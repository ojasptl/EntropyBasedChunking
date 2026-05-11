# EntropyBasedChunking

EntropyBasedChunking is a PDF-focused RAG (Retrieval-Augmented Generation) demo that combines:
- Entropy-aware sentence chunking
- Hybrid dense + sparse retrieval (Qdrant + BM25)
- Reciprocal Rank Fusion merging
- Cross-encoder reranking
- LLM answer generation

It ships with both a CLI workflow and a Streamlit UI for exploring PDFs interactively.

---

## What it does
1. Extracts text from a PDF
2. Splits text into semantically coherent chunks using entropy signals
3. Embeds and stores chunks in an in-memory Qdrant vector store
4. Builds a BM25 index over the same chunks
5. Retrieves relevant chunks using hybrid search + reranking
6. Generates a concise answer using a local LLM (Qwen2.5-3B-Instruct)

---

## Repository structure
- `main.py` — CLI entry point for PDF processing + Q&A
- `app.py` — Streamlit UI for interactive browsing and querying
- `chunker.py` — entropy-based chunking logic
- `retriever.py` — BM25 index, hybrid search, reranking
- `pdf_utils.py` — PDF text extraction + Qdrant storage
- `llm.py` — LLM initialization and answer generation
- `requirements.txt` — Python dependencies

---

## Setup

### Prerequisites
- Python and pip installed
- Internet access on first run to download models

### Install dependencies
```bash
pip install -r requirements.txt
```

---

## Run the CLI
```bash
python main.py
```
You will be prompted to:
1. Provide a PDF file path
2. Ask questions about the PDF
3. Optionally choose `top_k` results per query

---

## Trial run example
Example CLI session using the "Attention Is All You Need" PDF and a series of questions:
```text
Enter your question (or 'quit' to exit): What exact formula is used for scaled dot-product attention?
Top K results (default 5): 5

🔍 Searching for: 'What exact formula is used for scaled dot-product attention?'

Loading weights: 100%
 105/105 [00:00<00:00, 514.24it/s, Materializing param=classifier.weight]
BertForSequenceClassification LOAD REPORT from: cross-encoder/ms-marco-MiniLM-L-6-v2
Key                          | Status     |  | 
-----------------------------+------------+--+-
bert.embeddings.position_ids | UNEXPECTED |  | 

Notes:
- UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.

📑 Top 5 Results:

================================================================================

[Result 1] (Rerank score: 6.1258)
Chunk #45 | Sentences: 4 | Length: 451

3.2.1 Scaled Dot-Product Attention
We call our particular attention "Scaled Dot-Product Attention" (Figure 2). The input consists of
queries and keys of dimension dk, and values of dimension dv. We compute the dot products of the
query with all keys, divide each by√dk, and apply a softmax function to obtain the weights on the
values. In practice, we compute the attention function on a set of queries simultaneously, packed together
into a matrix Q.
--------------------------------------------------------------------------------

[Result 2] (Rerank score: 6.0962)
Chunk #47 | Sentences: 4 | Length: 472

The keys and values are also packed together into matrices KandV. We compute
the matrix of outputs as:
Attention( Q, K, V ) = softmax(QKT
√dk)V (1)
The two most commonly used attention functions are additive attention [ 2], and dot-product (multi-
plicative) attention. Dot-product attention is identical to our algorithm, except for the scaling factor
of1√dk. Additive attention computes the compatibility function using a feed-forward network with
a single hidden layer.
--------------------------------------------------------------------------------

[Result 3] (Rerank score: 5.7765)
Chunk #48 | Sentences: 3 | Length: 409

Dot-product attention is identical to our algorithm, except for the scaling factor
of1√dk. Additive attention computes the compatibility function using a feed-forward network with
a single hidden layer. While the two are similar in theoretical complexity, dot-product attention is
much faster and more space-efficient in practice, since it can be implemented using highly optimized
matrix multiplication code.
--------------------------------------------------------------------------------

[Result 4] (Rerank score: 5.5320)
Chunk #44 | Sentences: 4 | Length: 417

(right) Multi-Head Attention consists of several
attention layers running in parallel. of the values, where the weight assigned to each value is computed by a compatibility function of the
query with the corresponding key. 3.2.1 Scaled Dot-Product Attention
We call our particular attention "Scaled Dot-Product Attention" (Figure 2). The input consists of
queries and keys of dimension dk, and values of dimension dv.
--------------------------------------------------------------------------------

[Result 5] (Rerank score: 4.8824)
Chunk #51 | Sentences: 4 | Length: 676

While for small values of dkthe two mechanisms perform similarly, additive attention outperforms
dot product attention without scaling for larger values of dk[3]. We suspect that for large values of
dk, the dot products grow large in magnitude, pushing the softmax function into regions where it has
extremely small gradients4. To counteract this effect, we scale the dot products by1√dk. 3.2.2 Multi-Head Attention
Instead of performing a single attention function with dmodel-dimensional keys, values and queries,
we found it beneficial to linearly project the queries, keys and values htimes with different, learned
linear projections to dk,dkanddvdimensions, respectively.
--------------------------------------------------------------------------------

💡 Generating Answer...

================================================================================
Q: What exact formula is used for scaled dot-product attention?

A: The formula used for scaled dot-product attention is:

\[ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \]

Where \(Q\) represents the queries, \(K\) represents the keys, \(V\) represents the values, and \(d_k\) is the dimensionality of the keys and queries.
================================================================================

================================================================================

Enter your question (or 'quit' to exit): What learning rate schedule was used?
Top K results (default 5): 6

🔍 Searching for: 'What learning rate schedule was used?'

Loading weights: 100%
 105/105 [00:00<00:00, 479.56it/s, Materializing param=classifier.weight]
BertForSequenceClassification LOAD REPORT from: cross-encoder/ms-marco-MiniLM-L-6-v2
Key                          | Status     |  | 
-----------------------------+------------+--+-
bert.embeddings.position_ids | UNEXPECTED |  | 

Notes:
- UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.

📑 Top 6 Results:

================================================================================

[Result 1] (Rerank score: -2.2953)
Chunk #98 | Sentences: 3 | Length: 727

5.3 Optimizer
We used the Adam optimizer [ 20] with β1= 0.9,β2= 0.98andϵ= 10−9. We varied the learning
rate over the course of training, according to the formula:
lrate =d−0.5
model·min(step_num−0.5, step _num·warmup _steps−1.5) (3)
This corresponds to increasing the learning rate linearly for the first warmup _steps training steps,
and decreasing it thereafter proportionally to the inverse square root of the step number. We used
warmup _steps = 4000 . 5.4 Regularization
We employ three types of regularization during training:
7
Table 2: The Transformer achieves better BLEU scores than previous state-of-the-art models on the
English-to-German and English-to-French newstest2014 tests at a fraction of the training cost.
--------------------------------------------------------------------------------

[Result 2] (Rerank score: -2.8331)
Chunk #97 | Sentences: 3 | Length: 483

The big models were trained for 300,000 steps
(3.5 days). 5.3 Optimizer
We used the Adam optimizer [ 20] with β1= 0.9,β2= 0.98andϵ= 10−9. We varied the learning
rate over the course of training, according to the formula:
lrate =d−0.5
model·min(step_num−0.5, step _num·warmup _steps−1.5) (3)
This corresponds to increasing the learning rate linearly for the first warmup _steps training steps,
and decreasing it thereafter proportionally to the inverse square root of the step number.
--------------------------------------------------------------------------------

[Result 3] (Rerank score: -3.0296)
Chunk #99 | Sentences: 3 | Length: 1242

We varied the learning
rate over the course of training, according to the formula:
lrate =d−0.5
model·min(step_num−0.5, step _num·warmup _steps−1.5) (3)
This corresponds to increasing the learning rate linearly for the first warmup _steps training steps,
and decreasing it thereafter proportionally to the inverse square root of the step number. We used
warmup _steps = 4000 . 5.4 Regularization
We employ three types of regularization during training:
7
Table 2: The Transformer achieves better BLEU scores than previous state-of-the-art models on the
English-to-German and English-to-French newstest2014 tests at a fraction of the training cost. ModelBLEU Training Cost (FLOPs)
EN-DE EN-FR EN-DE EN-FR
ByteNet [18] 23.75
Deep-Att + PosUnk [39] 39.2 1.0·1020
GNMT + RL [38] 24.6 39.92 2.3·10191.4·1020
ConvS2S [9] 25.16 40.46 9.6·10181.5·1020
MoE [32] 26.03 40.56 2.0·10191.2·1020
Deep-Att + PosUnk Ensemble [39] 40.4 8.0·1020
GNMT + RL Ensemble [38] 26.30 41.16 1.8·10201.1·1021
ConvS2S Ensemble [9] 26.36 41.29 7.7·10191.2·1021
Transformer (base model) 27.3 38.1 3.3·1018
Transformer (big) 28.4 41.8 2.3·1019
Residual Dropout We apply dropout [ 33] to the output of each sub-layer, before it is added to the
sub-layer input and normalized.
--------------------------------------------------------------------------------

[Result 4] (Rerank score: -8.0314)
Chunk #107 | Sentences: 5 | Length: 445

The Transformer (big) model trained for English-to-French used
dropout rate Pdrop= 0.1, instead of 0.3. For the base models, we used a single model obtained by averaging the last 5 checkpoints, which
were written at 10-minute intervals. For the big models, we averaged the last 20 checkpoints. We
used beam search with a beam size of 4and length penalty α= 0.6[38]. These hyperparameters
were chosen after experimentation on the development set.
--------------------------------------------------------------------------------

[Result 5] (Rerank score: -8.3141)
Chunk #106 | Sentences: 3 | Length: 468

On the WMT 2014 English-to-French translation task, our big model achieves a BLEU score of 41.0,
outperforming all of the previously published single models, at less than 1/4the training cost of the
previous state-of-the-art model. The Transformer (big) model trained for English-to-French used
dropout rate Pdrop= 0.1, instead of 0.3. For the base models, we used a single model obtained by averaging the last 5 checkpoints, which
were written at 10-minute intervals.
--------------------------------------------------------------------------------

[Result 6] (Rerank score: -8.6906)
Chunk #105 | Sentences: 3 | Length: 518

Training took 3.5days on 8P100 GPUs. Even our base model
surpasses all previously published models and ensembles, at a fraction of the training cost of any of
the competitive models. On the WMT 2014 English-to-French translation task, our big model achieves a BLEU score of 41.0,
outperforming all of the previously published single models, at less than 1/4the training cost of the
previous state-of-the-art model. The Transformer (big) model trained for English-to-French used
dropout rate Pdrop= 0.1, instead of 0.3.
--------------------------------------------------------------------------------

💡 Generating Answer...

================================================================================
Q: What learning rate schedule was used?

A: The learning rate schedule used was defined by the formula:

\[ lrate = d^{-0.5} \cdot \text{model} \cdot \min(\text{step_num}-0.5, \text{step_num} \cdot \text{warmup_steps} - 1.5) \]

It increased linearly for the first \( \text{warmup_steps} \) training steps and then decreased proportionally to the inverse square root of the step number. Warmup_steps was set to 4000.
================================================================================

================================================================================

Enter your question (or 'quit' to exit): How many attention heads were used in the base Transformer model?
Top K results (default 5): 5

🔍 Searching for: 'How many attention heads were used in the base Transformer model?'

Loading weights: 100%
 105/105 [00:00<00:00, 362.62it/s, Materializing param=classifier.weight]
BertForSequenceClassification LOAD REPORT from: cross-encoder/ms-marco-MiniLM-L-6-v2
Key                          | Status     |  | 
-----------------------------+------------+--+-
bert.embeddings.position_ids | UNEXPECTED |  | 

Notes:
- UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.

📑 Top 5 Results:

================================================================================

[Result 1] (Rerank score: 5.2541)
Chunk #57 | Sentences: 3 | Length: 522

In this work we employ h= 8 parallel attention layers, or heads. For each of these we use
dk=dv=dmodel/h= 64 . Due to the reduced dimension of each head, the total computational cost
is similar to that of single-head attention with full dimensionality. 3.2.3 Applications of Attention in our Model
The Transformer uses multi-head attention in three different ways:
•In "encoder-decoder attention" layers, the queries come from the previous decoder layer,
and the memory keys and values come from the output of the encoder.
--------------------------------------------------------------------------------

[Result 2] (Rerank score: 4.7541)
Chunk #59 | Sentences: 3 | Length: 475

3.2.3 Applications of Attention in our Model
The Transformer uses multi-head attention in three different ways:
•In "encoder-decoder attention" layers, the queries come from the previous decoder layer,
and the memory keys and values come from the output of the encoder. This allows every
position in the decoder to attend over all positions in the input sequence. This mimics the
typical encoder-decoder attention mechanisms in sequence-to-sequence models such as
[38, 2, 9].
--------------------------------------------------------------------------------

[Result 3] (Rerank score: 4.5777)
Chunk #58 | Sentences: 3 | Length: 551

For each of these we use
dk=dv=dmodel/h= 64 . Due to the reduced dimension of each head, the total computational cost
is similar to that of single-head attention with full dimensionality. 3.2.3 Applications of Attention in our Model
The Transformer uses multi-head attention in three different ways:
•In "encoder-decoder attention" layers, the queries come from the previous decoder layer,
and the memory keys and values come from the output of the encoder. This allows every
position in the decoder to attend over all positions in the input sequence.
--------------------------------------------------------------------------------

[Result 4] (Rerank score: 1.3216)
Chunk #35 | Sentences: 4 | Length: 528

2
Figure 1: The Transformer - model architecture. The Transformer follows this overall architecture using stacked self-attention and point-wise, fully
connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1,
respectively. 3.1 Encoder and Decoder Stacks
Encoder: The encoder is composed of a stack of N= 6 identical layers. Each layer has two
sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position-
wise fully connected feed-forward network.
--------------------------------------------------------------------------------

[Result 5] (Rerank score: 1.2342)
Chunk #116 | Sentences: 0 | Length: 419

We present these results in Table 3. In Table 3 rows (A), we vary the number of attention heads and the attention key and value dimensions,
keeping the amount of computation constant, as described in Section 3.2.2. While single-head
attention is 0.9 BLEU worse than the best setting, quality also drops off with too many heads. In Table 3 rows (B), we observe that reducing the attention key size dkhurts model quality.
--------------------------------------------------------------------------------

💡 Generating Answer...

================================================================================
Q: How many attention heads were used in the base Transformer model?

A: The base Transformer model used 8 attention heads.
================================================================================

================================================================================

Enter your question (or 'quit' to exit): What dimensionality was used for embeddings in the base model?
Top K results (default 5): 5

🔍 Searching for: 'What dimensionality was used for embeddings in the base model?'

Loading weights: 100%
 105/105 [00:00<00:00, 427.82it/s, Materializing param=classifier.weight]
BertForSequenceClassification LOAD REPORT from: cross-encoder/ms-marco-MiniLM-L-6-v2
Key                          | Status     |  | 
-----------------------------+------------+--+-
bert.embeddings.position_ids | UNEXPECTED |  | 

Notes:
- UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.

📑 Top 5 Results:

================================================================================

[Result 1] (Rerank score: 4.2190)
Chunk #66 | Sentences: 4 | Length: 505

Another way of describing this is as two convolutions with kernel size 1. The dimensionality of input and output is dmodel = 512 , and the inner-layer has dimensionality
dff= 2048 . 3.4 Embeddings and Softmax
Similarly to other sequence transduction models, we use learned embeddings to convert the input
tokens and output tokens to vectors of dimension dmodel. We also use the usual learned linear transfor-
mation and softmax function to convert the decoder output to predicted next-token probabilities.
--------------------------------------------------------------------------------

[Result 2] (Rerank score: 1.7853)
Chunk #102 | Sentences: 4 | Length: 653

In addition, we apply dropout to the sums of the embeddings and the
positional encodings in both the encoder and decoder stacks. For the base model, we use a rate of
Pdrop= 0.1. Label Smoothing During training, we employed label smoothing of value ϵls= 0.1[36]. This
hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score. 6 Results
6.1 Machine Translation
On the WMT 2014 English-to-German translation task, the big transformer model (Transformer (big)
in Table 2) outperforms the best previously reported models (including ensembles) by more than 2.0
BLEU, establishing a new state-of-the-art BLEU score of 28.4.
--------------------------------------------------------------------------------

[Result 3] (Rerank score: 1.1402)
Chunk #67 | Sentences: 3 | Length: 463

3.4 Embeddings and Softmax
Similarly to other sequence transduction models, we use learned embeddings to convert the input
tokens and output tokens to vectors of dimension dmodel. We also use the usual learned linear transfor-
mation and softmax function to convert the decoder output to predicted next-token probabilities. In
our model, we share the same weight matrix between the two embedding layers and the pre-softmax
linear transformation, similar to [ 30].
--------------------------------------------------------------------------------

[Result 4] (Rerank score: 0.4312)
Chunk #74 | Sentences: 4 | Length: 514

That is, each dimension of the positional encoding
corresponds to a sinusoid. The wavelengths form a geometric progression from 2πto10000 ·2π. We
chose this function because we hypothesized it would allow the model to easily learn to attend by
relative positions, since for any fixed offset k,PEpos+kcan be represented as a linear function of
PEpos. We also experimented with using learned positional embeddings [ 9] instead, and found that the two
versions produced nearly identical results (see Table 3 row (E)).
--------------------------------------------------------------------------------

[Result 5] (Rerank score: 0.1883)
Chunk #72 | Sentences: 4 | Length: 488

To this end, we add "positional encodings" to the input embeddings at the
bottoms of the encoder and decoder stacks. The positional encodings have the same dimension dmodel
as the embeddings, so that the two can be summed. There are many choices of positional encodings,
learned and fixed [9]. In this work, we use sine and cosine functions of different frequencies:
PE(pos,2i)=sin(pos/100002i/d model)
PE(pos,2i+1)=cos(pos/100002i/d model)
where posis the position and iis the dimension.
--------------------------------------------------------------------------------

💡 Generating Answer...

================================================================================
Q: What dimensionality was used for embeddings in the base model?

A: The dimensionality used for embeddings in the base model was dmodel = 512.
================================================================================

================================================================================

Enter your question (or 'quit' to exit): Does the Transformer use CNN layers for sequence modeling?
Top K results (default 5): 5

🔍 Searching for: 'Does the Transformer use CNN layers for sequence modeling?'

Loading weights: 100%
 105/105 [00:00<00:00, 510.29it/s, Materializing param=classifier.weight]
BertForSequenceClassification LOAD REPORT from: cross-encoder/ms-marco-MiniLM-L-6-v2
Key                          | Status     |  | 
-----------------------------+------------+--+-
bert.embeddings.position_ids | UNEXPECTED |  | 

Notes:
- UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.

📑 Top 5 Results:

================================================================================

[Result 1] (Rerank score: 3.1494)
Chunk #133 | Sentences: 3 | Length: 552

In contrast to RNN sequence-to-sequence models [ 37], the Transformer outperforms the Berkeley-
Parser [29] even when training only on the WSJ training set of 40K sentences. 7 Conclusion
In this work, we presented the Transformer, the first sequence transduction model based entirely on
attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with
multi-headed self-attention. For translation tasks, the Transformer can be trained significantly faster than architectures based
on recurrent or convolutional layers.
--------------------------------------------------------------------------------

[Result 2] (Rerank score: 2.7547)
Chunk #59 | Sentences: 3 | Length: 475

3.2.3 Applications of Attention in our Model
The Transformer uses multi-head attention in three different ways:
•In "encoder-decoder attention" layers, the queries come from the previous decoder layer,
and the memory keys and values come from the output of the encoder. This allows every
position in the decoder to attend over all positions in the input sequence. This mimics the
typical encoder-decoder attention mechanisms in sequence-to-sequence models such as
[38, 2, 9].
--------------------------------------------------------------------------------

[Result 3] (Rerank score: 2.7054)
Chunk #134 | Sentences: 3 | Length: 498

7 Conclusion
In this work, we presented the Transformer, the first sequence transduction model based entirely on
attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with
multi-headed self-attention. For translation tasks, the Transformer can be trained significantly faster than architectures based
on recurrent or convolutional layers. On both WMT 2014 English-to-German and WMT 2014
English-to-French translation tasks, we achieve a new state of the art.
--------------------------------------------------------------------------------

[Result 4] (Rerank score: 1.9278)
Chunk #132 | Sentences: 0 | Length: 652

Our results in Table 4 show that despite the lack of task-specific tuning our model performs sur-
prisingly well, yielding better results than all previously reported models with the exception of the
Recurrent Neural Network Grammar [8]. In contrast to RNN sequence-to-sequence models [ 37], the Transformer outperforms the Berkeley-
Parser [29] even when training only on the WSJ training set of 40K sentences. 7 Conclusion
In this work, we presented the Transformer, the first sequence transduction model based entirely on
attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with
multi-headed self-attention.
--------------------------------------------------------------------------------

[Result 5] (Rerank score: 1.8469)
Chunk #123 | Sentences: 3 | Length: 429

Furthermore, RNN sequence-to-sequence
models have not been able to attain state-of-the-art results in small-data regimes [37]. We trained a 4-layer transformer with dmodel = 1024 on the Wall Street Journal (WSJ) portion of the
Penn Treebank [ 25], about 40K training sentences. We also trained it in a semi-supervised setting,
using the larger high-confidence and BerkleyParser corpora from with approximately 17M sentences
[37].
--------------------------------------------------------------------------------

💡 Generating Answer...

================================================================================
Q: Does the Transformer use CNN layers for sequence modeling?

A: No, the Transformer does not use CNN layers for sequence modeling. It replaces recurrent layers with multi-headed self-attention mechanisms.
================================================================================

================================================================================

Enter your question (or 'quit' to exit): Does the paper use GRUs in the encoder?
Top K results (default 5): 5

🔍 Searching for: 'Does the paper use GRUs in the encoder?'

Loading weights: 100%
 105/105 [00:00<00:00, 488.03it/s, Materializing param=classifier.weight]
BertForSequenceClassification LOAD REPORT from: cross-encoder/ms-marco-MiniLM-L-6-v2
Key                          | Status     |  | 
-----------------------------+------------+--+-
bert.embeddings.position_ids | UNEXPECTED |  | 

Notes:
- UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.

📑 Top 5 Results:

================================================================================

[Result 1] (Rerank score: -6.1241)
Chunk #0 | Sentences: 0 | Length: 740

Provided proper attribution is provided, Google hereby grants permission to
reproduce the tables and figures in this paper solely for use in journalistic or
scholarly works. Attention Is All You Need
Ashish Vaswani∗
Google Brain
avaswani@google.comNoam Shazeer∗
Google Brain
noam@google.comNiki Parmar∗
Google Research
nikip@google.comJakob Uszkoreit∗
Google Research
usz@google.com
Llion Jones∗
Google Brain
llion@google.comAidan N. Gomez∗ †
University of Toronto
aidan@cs.toronto.eduŁukasz Kaiser∗
Google Brain
lukaszkaiser@google.com
Illia Polosukhin∗ ‡
illia.polosukhin@gmail.com
Abstract
The dominant sequence transduction models are based on complex recurrent or
convolutional neural networks that include an encoder and a decoder.
--------------------------------------------------------------------------------

[Result 2] (Rerank score: -6.3433)
Chunk #72 | Sentences: 4 | Length: 488

To this end, we add "positional encodings" to the input embeddings at the
bottoms of the encoder and decoder stacks. The positional encodings have the same dimension dmodel
as the embeddings, so that the two can be summed. There are many choices of positional encodings,
learned and fixed [9]. In this work, we use sine and cosine functions of different frequencies:
PE(pos,2i)=sin(pos/100002i/d model)
PE(pos,2i+1)=cos(pos/100002i/d model)
where posis the position and iis the dimension.
--------------------------------------------------------------------------------

[Result 3] (Rerank score: -6.3584)
Chunk #39 | Sentences: 3 | Length: 414

In addition to the two
sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head
attention over the output of the encoder stack. Similar to the encoder, we employ residual connections
around each of the sub-layers, followed by layer normalization. We also modify the self-attention
sub-layer in the decoder stack to prevent positions from attending to subsequent positions.
--------------------------------------------------------------------------------

[Result 4] (Rerank score: -6.3609)
Chunk #60 | Sentences: 5 | Length: 657

This allows every
position in the decoder to attend over all positions in the input sequence. This mimics the
typical encoder-decoder attention mechanisms in sequence-to-sequence models such as
[38, 2, 9]. •The encoder contains self-attention layers. In a self-attention layer all of the keys, values
and queries come from the same place, in this case, the output of the previous layer in the
encoder. Each position in the encoder can attend to all positions in the previous layer of the
encoder. •Similarly, self-attention layers in the decoder allow each position in the decoder to attend to
all positions in the decoder up to and including that position.
--------------------------------------------------------------------------------

[Result 5] (Rerank score: -6.3948)
Chunk #40 | Sentences: 3 | Length: 438

Similar to the encoder, we employ residual connections
around each of the sub-layers, followed by layer normalization. We also modify the self-attention
sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This
masking, combined with fact that the output embeddings are offset by one position, ensures that the
predictions for position ican depend only on the known outputs at positions less than i.
--------------------------------------------------------------------------------

💡 Generating Answer...

================================================================================
Q: Does the paper use GRUs in the encoder?

A: No, the paper does not mention using GRUs (Gated Recurrent Units) in the encoder. It describes using positional encodings instead and mentions employing residual connections around each sub-layer, followed by layer normalization, without specifying any specific type of recurrent unit like GRU.
================================================================================
```

---

## Run the Streamlit app
```bash
streamlit run app.py
```
Steps:
1. Upload a PDF
2. Enter a question
3. Review retrieved chunks and the generated answer

---

## Notes
- The vector database is in-memory and resets each run.
- First run downloads models:
  - SentenceTransformer (`all-MiniLM-L6-v2`)
  - Cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`)
  - LLM (`Qwen2.5-3B-Instruct`)
- GPU is optional but recommended for faster inference.

---

## Troubleshooting
- If PDF extraction fails, verify the file is not scanned-only (image-only) content.
- NLTK will download `punkt_tab` automatically on first use if missing.
