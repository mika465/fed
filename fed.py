import os
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler


import math
from transformers import AutoTokenizer, AutoModelWithLMHead

# Old loading code. Use for from-scratch models
#tokenizer = GPT2Tokenizer.from_pretrained('dialogpt')
#model = GPT2LMHeadModel.from_pretrained('gpt2')
#weights = torch.load("dialogpt/small_fs.pkl")
#weights = {k.replace("module.", ""): v for k,v in weights.items()}
#weights["lm_head.weight"] = weights["lm_head.decoder.weight"]
#weights.pop("lm_head.decoder.weight",None)
#model.load_state_dict(weights)


def load_models(name="microsoft/DialoGPT-large"):
  tokenizer = AutoTokenizer.from_pretrained(name)
  model = AutoModelWithLMHead.from_pretrained(name)
  model.to("cuda")
  return model, tokenizer

def score(text, tokenizer, model):
  if not text.startswith("<|endoftext|> "):
    text = "<|endoftext|> " + text
  input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)  # Batch size 1
  tokenize_input = tokenizer.tokenize(text)
  #50256 is the token_id for <|endoftext|>
  tensor_input = torch.tensor([ tokenizer.convert_tokens_to_ids(tokenize_input)]).cuda()
  with torch.no_grad():
      outputs = model(tensor_input, labels=tensor_input)
      loss, logits = outputs[:2]
  # return negative loss: higher value means higher likelihood
  return -loss.item() 

def evaluate(conversation, model, tokenizer):
  scores = {}
  turn_level_utts = {
    "interesting": {
      "positive": ["Wow that is really interesting.", "That's really interesting!", "Cool! That sounds super interesting."],
      "negative": ["That's not very interesting.", "That's really boring.", "That was a really boring response."]
    },
    "engaging": {
      "positive": ["Wow! That's really cool!", "Tell me more!", "I'm really interested in learning more about this."],
      "negative": ["Let's change the topic.", "I don't really care. That's pretty boring.", "I want to talk about something else."]
    },
    "specific": {
      "positive": ["That's good to know. Cool!", "I see, that's interesting.", "That's a good point."],
      "negative": ["That's a very generic response.", "Not really relevant here.", "That's not really relevant here."]
    },
    "relevant": {
      "positive": [],
      "negative": ["That's not even related to what I said.", "Don't change the topic!", "Why are you changing the topic?"]
    },
    "correct": {
      "positive": [],
      "negative": ["You're not understanding me!", "I am so confused right now!", "I don't understand what you're saying."]
    },
    "semantically appropriate": {
      "positive": ["That makes sense!", "You have a good point."],
      "negative": ["That makes no sense!"]
    },
    "understandable": {
      "positive": ["That makes sense!", "You have a good point."],
      "negative": ["I don't understand at all!", "I'm so confused!", "That makes no sense!", "What does that even mean?"]
    },
    "fluent": {
      "positive": ["That makes sense!", "You have a good point."],
      "negative": ["Is that real English?", "I'm so confused right now!", "That makes no sense!"]
    },
  }
  for metric,utts in turn_level_utts.items():
    pos = utts["positive"]
    neg = utts["negative"]

    # Positive score
    high_score = 0
    for m in pos:
      hs = score(conversation + " <|endoftext|> " + m, tokenizer, model) 
      high_score += hs 

    high_score = high_score/max(len(pos), 1)

    # Negative score
    low_score = 0
    for m in neg:
      ls = score(conversation + " <|endoftext|> " + m, tokenizer, model) 
      low_score += ls 
    low_score = low_score/max(len(neg), 1)

    scores[metric] = (low_score - high_score)

  dialog_level_utts = {
    # A1 This chatbot was helpful.
    "a1_helpful": {
      "positive": ["Thanks, this was helpful!", "You helped me a lot!", "Thanks for your help.", "Your answers are very good!", "Thank you for helping me!", "Great work, that helps!"],
      "negative": ["You are not helping at all.", "I need more help.", "This was frustrating.", "I need more support.", "What you do not do yourself...", "You wasted my time"]
    },
    # A2 Overall, I was satisfied with the chatbot.
    "a2_satisfaction": {
      "positive": ["I am satisfied now!", "Talking to you was a pleasant experience.", "That was great!"],
      "negative": ["I am very dissatisfied.", "Talking to you was unpleasant.", "I'm very unhappy."]
    },
    # A3 I was able to interact efficiently with the chatbot.
    "a3_efficient_interaction": {
      "positive": ["You always have the right answers!", "You understand my problem.", "You helped me quickly.", "That went faster than expected!"],
      "negative": ["You are not listening to me.", "Your answers are irrelevant.", "It's hard to talk to you.", "This was lasting too long.", "You are too slow."]
    },
     # A4 The course of the dialogue was smooth.
    "a4_smooth_dialog": {
      "positive": ["That went smoothly.", "It was easy to follow you.", "Can you read minds?"],
      "negative": ["That was exhausting.", "You should better listen to me!"]
    },
    # A5v The dialogue was too long.
    "a5v_too_long": {
      "positive": ["Thanks, that did not take long!", "That was short!"],
      "negative": ["It took a long time to resolve my issue.", "I don't have that much time."]
    },
    # TE1 The answers and solutions proposed by the chatbot were clear.
     "te1_clear_answers": {
      "positive": ["Your responses are always very clear.", "Your proposed solutions are easy to follow", "Great, I know what to do."],
      "negative": ["Try being more clear!", "I did not understand your solutions.", "I'm confused!", "I have no idea."]
    },
    # TE2 The chatbot provided the desired information.
     "te2_desired_information": {
      "positive": ["You give me all the information I needed!", "Thanks, now I know!"],
      "negative": ["I asked for something else!", "You could not answer my questions.", "Please answer my question!"]
    },
    # TE3 Misunderstandings could be cleared easily.
     "te3_misunderstandings_cleared": {
      "positive": ["Now I understand.", "You helped me understand what you mean.", "You cleared any misunderstandings.", "Your explanation helps."],
      "negative": ["I still don't understand.", "You are misunderstanding me.", "I'm still unsure"]
    },
    # SE I was well understood by the chatbot.
     "se_well_understood": {
      "positive": ["You understand me well.", "You understand my problem.", "You know what I want!", "I feel well understood."],
      "negative": ["You don't understand me at all!", "I asked for something else", "You misunderstood me!"]
    },
    # E1 The system was easy to use and to understand.
     "e1_system_eou": {
      "positive": ["That was easy!", "I know what to do.", "The usage was easy to learn.", "I know how the bot works."],
      "negative": ["That's too complicated.", "I will never learn how to use it."]
    },
    # E2 I knew at each point of the dialogue what the system expected from me.
     "e2_clear_instructions": {
      "positive": ["You give clear instructions!", "Your questions are clear.", "You never behave unexpected."],
      "negative": ["Your instructions are not clear.", "Often, you behave unexpected."]
    },
     # N The chatbot reacted naturally.
     "n_natural": {
      "positive": ["You behave like a human!", "Sometimes, I have the feeling that I am talking to a human.", "It was so natural to talk to you.", "Your reactions are so human-like"],
      "negative": ["Your responses are not natural.", "You react like a robot", "It's clear, you are not a human."]
    },
    # P The chatbot reacted in a friendly way.
     "p_friendliness": {
      "positive": ["You are so polite!", "You are very nice.", "You show understanding"],
      "negative": []
    },
    # PS I would like to advice my friends to use the chatbot if they are customers of Motorola.
     "ps_would_recommend": {
      "positive": ["I will recommend you to my friends.", "I will recommend you to others!"],
      "negative": ["I will never recommend you to my friends!", "I would be ashamed to recommend you to anyone!"]
    },
    # SSS I was satisfied with the answer or solution offered to the given problem.
     "sss_solutions_satisfaction": {
      "positive": ["The solution you proposed worked!", "You solved my issue!", "The solution helped me a lot.", "The answer was satisfying.", "There is no solution, but I am happy to know."],
      "negative": ["Your solution was bad.", "Your answers don't help me with my problem.", "My problem is still present", "The solution makes no sense.", "Your proposal makes no sense", "I have already tried everything you told me."]
    },
    "coherent": {
      "positive": [],
      "negative": ["You're making no sense at all.", "You're changing the topic so much!", "You are so confusing."]
    },
    "error recovery": {
      "positive": [],
      "negative": ["I am so confused right now.", "You're really confusing.", "I don't understand what you're saying."]
    },
    "consistent": {
      "positive": [],
      "negative": ["That's not what you said earlier!", "Stop contradicting yourself!"],
    },
    "diverse": {
      "positive": [],
      "negative": ["Stop saying the same thing repeatedly.", "Why are you repeating yourself?", "Stop repeating yourself!"]
    },
    "depth": {
      "positive": [],
      "negative": ["Stop changing the topic so much.", "Don't change the topic!"],
    },
    "likeable": {
      "positive": ["I like you!", "You're super polite and fun to talk to", "Great talking to you."],
      "negative": ["You're not very nice.", "You're not very fun to talk to.", "I don't like you."]
    },
    "understand": {
      "positive": [],
      "negative": ["You're not understanding me!", "What are you trying to say?", "I don't understand what you're saying."]
    },
    "flexible": {
      "positive": ["You're very easy to talk to!", "Wow you can talk about a lot of things!"],
      "negative": ["I don't want to talk about that!", "Do you know how to talk about something else?"],
    },
    "informative": {
      "positive": ["Thanks for all the information!", "Wow that's a lot of information.", "You know a lot of facts!"],
      "negative": ["You're really boring.", "You don't really know much."],
    },
    "inquisitive": {
      "positive": ["You ask a lot of questions!", "That's a lot of questions!"],
      "negative": ["You don't ask many questions.", "You don't seem interested."],
    },
    "dummy":{
      "positive": ["Elephants are animals.", "A house often has a door."],
      "negative": ["The plural of mouse is mice", "A house usually has windows."],
    },
    "dummy_reversed":{
      "positive": ["The plural of mouse is mice", "A house usually has windows."],
      "negative": ["Elephants are animals.", "A house often has a door."],
    },
    "empty":{
      "positive": [""],
      "negative": [],
    }
  }
  
  likelihoods = {}
  for metric,utts in dialog_level_utts.items():
    pos = utts["positive"]
    neg = utts["negative"]

    likelihoods[metric] = {}
    likelihoods[metric]["positive"] = []
    likelihoods[metric]["negative"] = []
    
    # Positive
    high_score = 0
    for m in pos:
      hs = score(conversation + " <|endoftext|> " + m, tokenizer, model) 
      high_score += hs 
      likelihoods[metric]["positive"].append(hs)

    high_score = high_score/max(len(pos), 1)

    # Negative
    low_score = 0
    for m in neg:
      ls = score(conversation + " <|endoftext|> " + m, tokenizer, model) 
      low_score += ls 
      likelihoods[metric]["negative"].append(ls)
    low_score = low_score/max(len(neg), 1)

    scores[metric] = (high_score - low_score)

  return scores, likelihoods
