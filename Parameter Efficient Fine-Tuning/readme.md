# Parameter Efficicent Fine-Tuning
- [Overview](#overview)
- [what is PEFT ?](#parameter-efficient-fine-tuning-peft)
- [Practical Use-case](#practical-use-case)
- [PEFT Methods](#peft-methods)
    - [1. Prompt Modifications](#1-prompt-modifications)
        - [Soft Prompt Tuning](#soft-prompt-tuning)
        - [Soft Prompt Tuning vs Prompt Engineering](#soft-prompt-vs-prompting)
        - [Prefix Tuning](#prefix-tuning)



## Overview:
- Fine-tuning of large pre-trained models on downstream tasks is called “transfer learning”.
- While full fine-tuning pre-trained models on downstream tasks is a common, effective approach, it is an inefficient approach to transfer learning.
- The simplest way out for efficient fine-tuning could be to freeze the networks’ lower layers and adapt only the top ones to specific tasks.
- In this article, we’ll explore Parameter Efficient Fine-Tuning (PEFT) methods that enable us to adapt a pre-trained model to downstream tasks more efficiently – in a way that trains lesser parameters and hence saves cost and training time, while also yielding performance similar to full fine-tuning.

## Parameter-Efficient Fine-Tuning (PEFT)
- Parameter-Efficient Fine-Tuning (PEFT) in the context of Large Language Models (LLMs) refers to a set of techniques used to fine-tune a pre-trained model on specific tasks or datasets while only updating a small subset of the model's parameters. This approach is aimed at reducing the computational cost and memory requirements associated with training large models. Instead of updating all the parameters, PEFT methods typically involve modifying only a few strategic parameters or adding small, trainable modules to the pre-existing network, thereby preserving the general capabilities of the model while adapting it to new tasks. This makes the fine-tuning process more efficient and accessible, especially for applications with limited resources.
- The challenge is this: modern pre-trained models (like BERT, GPT, T5, etc.) contain hundreds of millions, if not billions, of parameters. Fine-tuning all these parameters on a downstream task, especially when the available dataset for that task is small, can easily lead to overfitting. The model may simply memorize the training data instead of learning genuine patterns. Moreover, introducing additional layers or parameters during fine-tuning can drastically increase computational requirements and memory consumption.
- PEFT allows to only fine-tune a small number of model parameters while freezing most of the parameters of the pre-trained LLM.

## Practical Use-case
- PEFT obviates the need for 40 or 80GB A100s to make use of powerful LLMs. In other words, you can fine-tune 10B+ parameter LLMs for your desired task for free or on cheap consumer GPUs.
- Using PEFT methods like LoRA, especially 4-bit quantized base models via QLoRA, you can fine-tune 10B+ parameter LLMs that are 30-40GB in size on 16GB GPUs. 
- If you’re fine-tuning on a single task, the base models are already so expressive that you need only a few (~10s-100s) of examples to perform well on this task. With PEFT via LoRA, you need to train only a trivial fraction (in this case, 0.08%), and though the weights are stored as 4-bit, computations are still done at 16-bit.
- ***Key takeaway:*** You can fine-tune powerful LLMs to perform well on a desired task using free compute. Use a <10B parameter model, which is still huge, and use quantization, PEFT, checkpointing, and provide a small training set, and you can quickly fine-tune this model for your use case.

## PEFT Methods
![](./image/1.webp)

### 1. Prompt Modifications:
#### **Soft Prompt Tuning:**
- First introduced in the [The Power of Scale for Parameter-Efficient Prompt Tuning](https://aclanthology.org/2021.emnlp-main.243.pdf); this paper by Lester et al. introduces a simple yet effective method called soft prompt tuning, which prepends a trainable tensor to the model’s input embeddings, essentially creating a soft prompt to condition frozen language models to perform specific downstream tasks. Unlike the discrete text prompts, soft prompts are learned through backpropagation and can be fine-tuned to incorporate signals from any number of labeled examples.
- Soft prompt tuning only requires storing a small task-specific prompt for each task, and enables mixed-task inference using the original pre-trained model.
- The authors show that prompt tuning outperforms few-shot learning by a large margin, and becomes more competitive with scale.
- This is an interesting approach that can help to effectively use a single frozen model for multi-task serving.
- Model tuning requires making a task-specific copy of the entire pre-trained model for each downstream task and inference must be performed in separate batches. Prompt tuning only requires storing a small task-specific prompt for each task, and enables mixed-task inference using the original pretrained model. With a T5 “XXL” model, each copy of the tuned model requires 11 billion parameters. By contrast, our tuned prompts would only require 20,480 parameters per task—a reduction of over five orders of magnitude – assuming a prompt length of 5 tokens.
- Thus, instead of using discrete text prompts, prompt tuning employs soft prompts. Soft prompts are learnable and conditioned through backpropagation, making them adaptable for specific tasks.

![](./image/2.jpg)

<br />

### **In Other words:**
Soft prompts are a way to tell a large language model (LLM) what to do, but without using any words. Instead, the LLM is trained on a set of examples, and then learns to recognize the patterns in those examples. These patterns are then used to create a soft prompt, which is a string of numbers that represents the patterns.

___soft prompts are a powerful tool for adapting LLMs to new tasks, especially for tasks where there is limited training data available.___

Esxample:
Imagine that you want to train an LLM to write poems. You could start by giving the LLM a set of example poems. The LLM would then learn to recognize the patterns in those poems, such as the rhyme scheme, the meter, and the subject matter. 

<br />

Once the LLM has learned the patterns in the example poems, you could create a soft prompt by extracting the patterns from the examples. This soft prompt could then be used to guide the LLM to write its own poems.

<br />

The LLM would not be able to read the soft prompt, but it would be able to recognize the patterns in the soft prompt and use those patterns to generate creative text.

The format of a soft prompt for the above example could be a string of numbers that represent the rhyme scheme, the meter, and the subject matter of the example poems. For example, the following soft prompt could be used to guide the LLM to write a poem about love:

`[ABAB rhyme scheme], [iambic tetrameter], [love]`

This soft prompt tells the LLM to write a poem with an ABAB rhyme scheme, in iambic tetrameter, and about the subject of love.

<br />

### Soft Prompt vs. Prompting:

**Prompting:** 
- involves creating specific input prompts to guide the model's responses. These prompts are non-trainable and consist solely of plain text that contextualizes or specifies the task for the model.

- Used to elicit specific types of responses from a pre-trained model without modifying the model itself.

**Advantages:**
- No additional training required.
- Can be quickly implemented for various tasks.
- Highly flexible.

**Disadvantages**
- Effectiveness depends on the skill of the person crafting the prompts.
- Results may lack consistency.
- Often requires trial and error to find effective prompts.

<br />

**Soft Prompt Tuning:**
- involves appending trainable parameters (soft prompts) to the model's input, which are optimized during a training phase to improve performance on specific tasks.

- Adapts a model to new tasks or enhances its performance in specific domains while keeping the core model weights unchanged.

**Advantages:**
- More systematic than manual prompting.
- The model learns to perform tasks during the tuning process, potentially leading to better performance.

**Disadvantages:**
- Requires computational resources for training the soft prompts.
- The effectiveness is contingent upon the number of trainable parameters and the initial quality of the model.

---- 

In Prompt Engineering we work on the language of our prompt to achieve better completion results. This process requires manual work and the prompt is limited by context window size.

in Soft Prompt Tuning, we add a trainable token to the input prompt. The added prompt must be same length as the original language token provided to the LLM. Then, the supervised learning algorithm trains the newly added prompts. In many cases, 20–100 tokens are enough to achieve a performance boost. It is important to note that the original weights of model are frozen, and only the new prompts are trained as part of this process.

--- 
- Soft prompt tuning and prompting a model with extra context are both methods designed to guide a model’s behavior for specific tasks, but they operate in different ways. Here’s how they differ:

1. **Mechanism:**
    - **Soft Prompt Tuning:** This involves introducing trainable parameters (soft prompts) that are concatenated or added to the model’s input embeddings. These soft prompts are learned during the fine-tuning process and are adjusted through backpropagation to condition the model to produce desired outputs for specific tasks.
    - **Prompting with Extra Context:** This method involves feeding the model with handcrafted or predefined text prompts that provide additional context. There’s no explicit fine-tuning; instead, the model leverages its pre-trained knowledge to produce outputs based on the provided context. This method is common in few-shot learning scenarios where the model is given a few examples as prompts and then asked to generalize to a new example.
2. **Trainability:**
    - **Soft Prompt Tuning:** The soft prompts are trainable. They get adjusted during the fine-tuning process to optimize the model’s performance on the target task.
    - **Prompting with Extra Context:** The prompts are static and not trainable. They’re designed (often manually) to give the model the necessary context for the desired task.
3. **Use Case:**
    - **Soft Prompt Tuning:** This method is particularly useful when there’s a need to adapt a pre-trained model to various downstream tasks without adding significant computational overhead. Since the soft prompts are learned and optimized, they can capture nuanced information necessary for the task.
    - **Prompting with Extra Context:** This is often used when fine-tuning isn’t feasible or when working with models in a zero-shot or few-shot setting. It’s a way to leverage the vast knowledge contained in large pre-trained models by just guiding their behavior with carefully crafted prompts.

<br />

### Prefix Tuning:
- Proposed in [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190), prefix-tuning is a lightweight alternative to fine-tuning for natural language generation tasks, which keeps language model parameters frozen, but optimizes a small continuous task-specific vector (called the prefix).
- Instead of adding a soft prompt to the model input, it prepends trainable parameters to the hidden states of all transformer blocks. During fine-tuning, the LM’s original parameters are kept frozen while the prefix parameters are updated.
- Prefix-tuning draws inspiration from prompting, allowing subsequent tokens to attend to this prefix as if it were “virtual tokens”.
- The figure below from the paper shows that fine-tuning (top) updates all Transformer parameters (the red Transformer box) and requires storing a full model copy for each task. They propose prefix-tuning (bottom), which freezes the Transformer parameters and only optimizes the prefix (the red prefix blocks). Consequently, prefix-tuning only need to store the prefix for each task, making prefix-tuning modular and space-efficient. Note that each vertical block denote transformer activations at one time step.

![3](./image/3.jpg)

---

### **In Other Words:**
- Prefix tuning is a technique aiming to streamline the process. Instead of relying on manual prompt engineering, it focuses on learning a continuous prompt that can be seamlessly optimized end-to-end. This learned prompt, when added to the model’s input, acts as a guiding beacon, providing the necessary context to steer the model’s behavior in alignment with the specific task at hand. It’s like giving the model a customized set of instructions without the hassle of intricate manual tweaking, making the entire process more efficient and dynamic. It also doesn’t require training multiple parameters from the model, training only less than 1000× the parameters of the model.
- prefix tuning prepends a learned continuous vector to the input. For example, in summarization, a prefix would be prepended to the input document. The prefix is tuned to steer the model to perform summarization while keeping the large pretrained model fixed. This is much more efficient, requiring tuning only 0.1% of the parameters compared to full fine-tuning.
- Prefix tuning draws inspiration from prompting methods like in GPT-3, but optimizes a continuous prefix vector rather than using discrete tokens. The paper shows prefix tuning can match the performance of full fine-tuning on table-to-text and summarization tasks, while using 1000x fewer parameters per task.

#### How Prefix Tuning works:
Prefix Tuning essentially prepends a learned continuous vector, called the prefix, to the input of the pretrained model.

<br />

Let’s take an example. Imagine we are prefix-tuning a Large Language Model (LLM) for Hate Speech Classification. The model takes an input x tweet and generates an output y which is the classification “Hate” or “Non-Hate”.

<br />

In prefix tuning, we’re doing a simple yet clever move — mixing x and y into a single sequence, let’s call it z = [x; y]. Why? Well, this combo creates a kind of “encoder-like” function. It’s super handy for tasks where y depends on x. It’s called Conditional Generation. This way, the model can smoothly go back and forth between x and y using its self-attention skills.
Moving along in the process, we introduce a prefix vector, let’s call it u, which is placed at the beginning of our sequence z, resulting in the concatenated form [u; x; y].

<br />

The prefix vector u is a matrix with dimensions (prefix_length × d), where d denotes the hidden dimension size. To put it into perspective, consider a scenario with a prefix length of 10 and a hidden size of 1024. In this case, the prefix would house a total of 10,240 tunable parameters.

This unified sequence is then systematically input into the Transformer model in an autoregressive manner. The model engages in attentive computations, focusing on prior tokens within the sequence z to predict the subsequent token. Specifically, the model computes hi, representing the current hidden state, as a function of zi and the past activations within its left context. This approach ensures the Transformer’s ability to progressively anticipate the upcoming tokens in the sequence.

![4](./image/4.webp)

--- 

- **Embedding Modification:** In Prefix Tuning, a series of task-specific vectors, or "prefixes," are prepended to the input embeddings of the sequence that is fed into the model. These prefixes are learnable parameters that are optimized during training.
- **Task Adaptation:** By training these prefix embeddings, the model can adapt to a new task without altering its core architecture or the bulk of its pre-trained weights. The trained prefixes essentially guide the model's attention and processing pathways to generate more appropriate responses for the specific task.

### **Usage:**
- **Fine-Tuning Alternative:** Prefix Tuning is used as an alternative to full model fine-tuning when there are constraints on computational resources or when model stability must be maintained across updates.
- **Specialized Tasks:** It is particularly useful for specialized tasks where only a subset of the model's behavior needs to be modified, such as task-specific classification, generation tasks, or adapting to new domains with limited data.

### **Advantages:**
- **Efficiency:** It requires less memory and computational power compared to fine-tuning the entire model, as only a small number of parameters are trained.
- __Flexibility:__ Prefixes can be easily swapped out for different tasks without interfering with the model's underlying capabilities.
- __Preservation of Generalization:__ By keeping most of the model's weights fixed, Prefix Tuning preserves the generalization abilities learned during pre-training.

### __Disadvantages__:
- **Limited Scope:** Since only a small part of the model is adapted, the changes it can make are less dramatic than those possible through full model fine-tuning.
- **Dependency on Pre-Trained Model Quality:** The effectiveness of Prefix Tuning heavily depends on the quality and versatility of the underlying pre-trained model.