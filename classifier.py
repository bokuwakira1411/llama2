
import torch
import torch.nn.functional as F

# change it with respect to the original model
from config import LlamaConfig
from llama import load_pretrained
from tokenizer import Tokenizer

class LlamaZeroShotClassifier(torch.nn.Module):
	def __init__(self, config: LlamaConfig, tokenizer: Tokenizer, label_names: list[str]):
		super(LlamaZeroShotClassifier, self).__init__()
		self.num_labels = config.num_labels
		self.llama = load_pretrained(config.pretrained_model_path)
		# Zero-shot classification does not require updating llama paramters.
		for param in self.llama.parameters():
			param.requires_grad = False
		assert len(label_names) == self.num_labels
		self.tokenizer = tokenizer
		self.label_name_ids = [tokenizer.encode(label, bos=False, eos=False) for label in label_names]


	def forward(self, input_ids):
		# compute the completion probability of each label string
		logits, _ = self.llama(input_ids)
		log_probabilities = F.log_softmax(logits, dim=-1)
		label_probabilities = torch.zeros((log_probabilities.shape[0], self.num_labels), device=log_probabilities.device)
		for i, label_token_ids in enumerate(self.label_name_ids):
			total_log_prob = torch.sum(log_probabilities[:, :, label_token_ids], axis=-1)
			label_probabilities[:, i] = total_log_prob[:, 0]
		return label_probabilities

class LlamaEmbeddingClassifier(torch.nn.Module):
	def __init__(self, config):
		super(LlamaEmbeddingClassifier, self).__init__()
		self.num_labels = config.num_labels
		self.llama = load_pretrained(config.pretrained_model_path)
		# If we use pretrain mode, we freeze Llama parameters.
		for param in self.llama.parameters():
			if config.option == 'pretrain':
				param.requires_grad = False
			elif config.option == 'finetune':
				param.requires_grad = True

		self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
		self.classifier_head = torch.nn.Linear(self.llama.config.dim, self.num_labels)

	def forward(self, input_ids):
		'''
		1) Find the hidden state after the final token of the input sequence
		2) Apply dropout (self.dropout) to the hidden state at training time to mitigate
		   overfitting.
		2) Pass this through the classifier head (self.classifier_head), which will return
		   logits (unnormalized probabilities) over all classes.
		3) Take the log-softmax of the logits and return log-probabilities over all classes.
		'''
		# todo
		# raise NotImplementedError
		logits, hidden_states = self.llama(input_ids)
		last_hidden = hidden_states[:,-1,:]
		dropped = self.dropout(last_hidden)
		logits = self.classifier_head(dropped)
		log_probs = F.log_softmax(logits, dim=-1)
		return log_probs
# the best for testing cfimdb-test.txt
# import torch
# import torch.nn.functional as F

# from config import LlamaConfig
# from llama import load_pretrained
# from tokenizer import Tokenizer


# class LlamaZeroShotClassifier(torch.nn.Module):
#     def __init__(self, config: LlamaConfig, tokenizer: Tokenizer, label_names: list[str]):
#         super().__init__()
#         self.num_labels = config.num_labels
#         self.llama = load_pretrained(config.pretrained_model_path)

#         for param in self.llama.parameters():
#             param.requires_grad = False

#         assert len(label_names) == self.num_labels
#         self.tokenizer = tokenizer
#         self.label_name_ids = [tokenizer.encode(label, bos=False, eos=False) for label in label_names]

#     def forward(self, input_ids):
#         logits, _ = self.llama(input_ids)
#         log_probs = F.log_softmax(logits, dim=-1)

#         B, T, V = log_probs.shape
#         label_scores = torch.zeros((B, self.num_labels), device=log_probs.device)

#         for i, label_token_ids in enumerate(self.label_name_ids):
#             score = torch.zeros(B, device=log_probs.device)

#             for j, token_id in enumerate(label_token_ids):
#                 pos = T - len(label_token_ids) + j
#                 score += log_probs[:, pos, token_id]

#             label_scores[:, i] = score

#         return label_scores


# class LlamaEmbeddingClassifier(torch.nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.num_labels = config.num_labels
#         self.llama = load_pretrained(config.pretrained_model_path)

#         for param in self.llama.parameters():
#             if config.option == 'pretrain':
#                 param.requires_grad = False
#             elif config.option == 'finetune':
#                 param.requires_grad = True

#         self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

#         self.classifier_head = torch.nn.Sequential(
#             torch.nn.Linear(self.llama.config.dim, self.llama.config.dim),
#             torch.nn.ReLU(),
#             torch.nn.Dropout(config.hidden_dropout_prob),
#             torch.nn.Linear(self.llama.config.dim, self.num_labels)
#         )

#     def forward(self, input_ids):
#         logits, hidden_states = self.llama(input_ids)

#         pooled = hidden_states.mean(dim=1)
#         dropped = self.dropout(pooled)

#         logits = self.classifier_head(dropped)
#         log_probs = F.log_softmax(logits, dim=-1)

#         return log_probs

# the best for testing sst-dev.txt
# import torch
# import torch.nn.functional as F

# from config import LlamaConfig
# from llama import load_pretrained
# from tokenizer import Tokenizer


# class LlamaZeroShotClassifier(torch.nn.Module):
#     def __init__(self, config: LlamaConfig, tokenizer: Tokenizer, label_names: list[str]):
#         super().__init__()
#         self.num_labels = config.num_labels
#         self.llama = load_pretrained(config.pretrained_model_path)

#         for param in self.llama.parameters():
#             param.requires_grad = False

#         assert len(label_names) == self.num_labels
#         self.tokenizer = tokenizer
#         self.label_name_ids = [
#             tokenizer.encode(label, bos=False, eos=False)
#             for label in label_names
#         ]

#     def forward(self, input_ids):
#         logits, _ = self.llama(input_ids)
#         log_probs = F.log_softmax(logits, dim=-1)

#         B, T, V = log_probs.shape
#         label_scores = torch.zeros((B, self.num_labels), device=log_probs.device)

#         for i, label_token_ids in enumerate(self.label_name_ids):
#             score = torch.zeros(B, device=log_probs.device)

#             for j, token_id in enumerate(label_token_ids):
#                 pos = T - len(label_token_ids) + j
#                 score += log_probs[:, pos, token_id]

#             score = score / len(label_token_ids)

#             label_scores[:, i] = score

#         return label_scores

# class LlamaEmbeddingClassifier(torch.nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.num_labels = config.num_labels
#         self.llama = load_pretrained(config.pretrained_model_path)

#         for param in self.llama.parameters():
#             if config.option == 'pretrain':
#                 param.requires_grad = False
#             elif config.option == 'finetune':
#                 param.requires_grad = True

#         dim = self.llama.config.dim

#         self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

#         self.fc1 = torch.nn.Linear(dim * 2, dim)
#         self.fc2 = torch.nn.Linear(dim, dim)
#         self.out = torch.nn.Linear(dim, self.num_labels)

#         self.activation = torch.nn.GELU()

#     def forward(self, input_ids):
#         logits, hidden_states = self.llama(input_ids)

#         mean_pool = hidden_states.mean(dim=1)
#         max_pool, _ = hidden_states.max(dim=1)

#         pooled = torch.cat([mean_pool, max_pool], dim=-1)

#         x = self.dropout(pooled)

#         h = self.activation(self.fc1(x))
#         h = self.dropout(h)

#         h2 = self.activation(self.fc2(h))
#         h = h + h2   

#         logits = self.out(h)

#         return F.log_softmax(logits, dim=-1)
