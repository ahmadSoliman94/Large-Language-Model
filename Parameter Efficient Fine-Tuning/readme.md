# Parameter Efficicent Fine-Tuning
- [Overview](#overview)
- [what is PEFT ?](#parameter-efficient-fine-tuning-peft)
- [Practical Use-case](#practical-use-case)
- [PEFT Methods](#peft-methods)
    - [1. Prompt Modifications](#1-prompt-modifications)
        - [Soft Prompt Tuning](#soft-prompt-tuning)
        - [Soft Prompt Tuning vs Prompt Engineering](#soft-prompt-vs-prompting)



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

#### - Soft Prompt vs. Prompting:

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




