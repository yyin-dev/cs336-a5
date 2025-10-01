## Problem (math_baseline)

(a) `math_baseline.py`. I evaluated on GSM8k and MATH test set, using `r1_zero_reward_function`.

* MATH:

  * format=1, answer=1, reward=1: 12

  * format=1, answer=0, reward=0: 62

  * format=0, answer=0, reward=0: 426.

(b) Observed cases where the format reward is zero: 

* the grader requires both `</think> <answer>` and `</answer>`, but the model doesn't always do. Sometimes it forgot to close with `</answer>`, sometimes it output the answer doesn't without closing `</think>`. However, the model could be outputting correct answers.
* the model doesn't finish answer with in the answer token count. 

```
{"question": "A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?", "truth": "It takes 2/2=<<2/2=1>>1 bolt of white fiber\nSo the total amount of fabric is 2+1=<<2+1=3>>3 bolts of fabric\n#### 3", "response": "initial`` The problem can be broken down as follows:\n\n1. Determine how much white fiber is required based on the amount of blue fiber.\n2. Sum the amount of blue and white fibers to get the total amount required.\n\nHere's the step-by-step reasoning and the corresponding Python code to solve the problem.\n\n1. The problem states that the robe takes 2 bolts of blue fiber.\n2. It also states that it takes half the amount of white fiber compared to blue fiber. So, if the blue fiber is 2 bolts, the white fiber will be \\( \\frac{2}{2} = 1 \\) bolt.\n3. The total amount of fiber required is the sum of the blue and white fibers: \\( 2 + 1 = 3 \\) bolts.\n\nThe Python code to solve this problem is straightforward:\n\n```python\n# Define the amount of blue fiber\nblue_fiber = 2\n\n# Calculate the amount of white fiber\nwhite_fiber = blue_fiber / 2\n\n# Calculate the total amount of fiber required\ntotal_fiber = blue_fiber + white_fiber\n\n# Print the total fiber required\nprint(total_fiber)\n```\n```output\n3.0\n```\nThe total amount of fiber required to make the robe is \\( \\boxed{3} \\) bolts.", "reward": {"format_reward": 0.0, "answer_reward": 0.0, "reward": 0.0}}
```

I think the issue is with the model.

Observed cases where the format reward is 1 but the answer reward is 0: I observed 10 cases; 2 out of 10 the model output correct answer, but still the answer reward is 0. This is problem with the parser. 

```
{"question": "Peter plans to go to the movies this week. He always gets a ticket for $7 and popcorn for $7. If he has 42 dollars for the week, how many times can he go to the movies?", "truth": "It costs him $14 to go to the movies because 7 + 7 = <<7+7=14>>14\nHe can go to the movies 3 times because 42 / 14 = <<42/14=3>>3\n#### 3", "response": " Peter can buy 3 times the movies for 42 dollars. </think> <answer> 3 </answer>", "reward": {"format_reward": 1.0, "answer_reward": 1.0, "reward": 1.0}}

{"question": "A wooden bridge can carry no more than 5000 pounds. A delivery truck filled with identical boxes, each weighing 15 pounds, will pass over the bridge. The combined weight of the driver and the empty truck is 3755 pounds. What is the maximum number of boxes which can be loaded onto the truck while not exceeding the bridge's weight limit?", "truth": "The boxes can weigh up to 5000 pounds - 3755 pounds = <<5000-3755=1245>>1245 pounds in total.\nThere can be 1245 / 15 = <<1245/15=83>>83 boxes loaded onto the truck without exceeding the bridge's weight limit.\n#### 83", "response": " The truck's weight without boxes is 3755 pounds. The remaining capacity of the bridge is 5000 - 3755 = 1245 pounds. Each box weighs 15 pounds, so the maximum number of boxes that can be loaded is 1245 / 15 = 83 boxes. </think> <answer> 83 boxes (answer format) </answer>", "reward": {"format_reward": 1.0, "answer_reward": 0.0, "reward": 0.0}}
```

(c) On MATH test set:

* Format correct ratio: 207/1200=17.3%
* Answer correct ration: 34/1200=2.83%.

## problem (compute_entropy)

Starting with softmax:
$$
p(x) = \frac{e^{\text{logits}(x)}}{\sum_{x \in X} e^{\text{logits}(x)}}
$$
Taking the logarithm:
$$
\log p(x) = \log \frac{e^{\text{logits}(x)}}{\sum_{x \in X} e^{\text{logits}(x)}} = \log e^{\text{logits}(x)} - \log \sum_{x \in X} e^{\text{logits}(x)} = \text{logits}(x) - \text{logsumexp}(x)
$$
Therefore:
$$
\boxed{\log p(x) = x - \text{logsumexp}(x)}
$$
And equivalently:
$$
\boxed{p(x) = e^{(x - \text{logsumexp}(x))}}
$$
Thus, 
$$
p(x)log(p(x)) = e^{(x - \text{logsumexp}(x))} (x - \text{logsumexp}(x))
$$

## problem (sft_microbatch_train_step)

What's the loss function in SFT? Why do we use "negative log likelihood "?

* "likelihood" is just probability. LLM products probabilities at each generation step. The goal is to maximize the probability that the model generates the label response, which is the product of the probabilities that the model generates each token in the label.
* "negative": Because gradient descent minimizes the loss while we want to maximize the probability, we use the negative probability as loss. 
* "log": The process of multiplying a sequence of probabilities is numerically unstable and can cause underflow, using log probabilities turns multiplication into sum. 

How is cross-entropy loss related to negative log likelihood? Cross-entropy loss in this case is another name for negative log likelihood. Cross-entropy measures the difference between two distributions. When you apply cross-entropy loss with a "groud-truth" distribution that's a one-hot vector, the formula simplifies to negative log likelihood.

## problem (sft_experiment)

![image-20250913143816985](https://raw.githubusercontent.com/yyin-dev/image_cloud/main/Picsee/image-20250913143816985_GQ5vNF.jpeg)

### Prompting

We are using r1-zero prompt: 

```
A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>
```

The training data looks like:

```
{'problem': 'How many units long is a segment whose endpoints are $(-4,1)$ and $(1,13)$?', 
 'solution': 'We use the distance formula: $\\sqrt{(-4 - 1)^2 + (1 - 13)^2},$ which is $\\sqrt{25 + 144} = \\sqrt{169} = \\boxed{13}$.\n\n- OR -\n\nWe note that the points $(-4,1)$, $(1,13)$, and $(1,1)$ form a right triangle with legs of length 5 and 12. $(5,12,13)$ is a Pythagorean triple, so the hypotenuse has length $\\boxed{13}$.', 
 'answer': '13', 
 'subject': 'Algebra', 
 'level': 2, 
 'unique_id': 'test/algebra/1570.json', 
 'gold_solution_steps': ['We use the distance formula:', '$\\sqrt{(-4 - 1)^2 + (1 - 13)^2},$ which is $\\sqrt{25 + 144} = \\sqrt{169} = \\boxed{13}$.', '- OR - We note that the points $(-4,1)$, $(1,13)$,', 'and $(1,1)$ form a right triangle with legs of length 5 and 12. $(5,12,13)$ is a Pythagorean triple, so the hypotenuse has length $\\boxed{13}$.']
}
```

Clearly, we should format the `problem` using the prompt template above. 

```
prompt = generate_prompt(R1_ZERO_PROMPT, data["problem"])
label = data["solution"]
```

However, the model is not learning.. Why?

If you examine the "solution" closely, you would see that it's not following the expected format in r1-zero prompt, thus the model is not learning to generate text in the right format. The grader script wouldn't even check answer correctness if the format is wrong, so the reward doesn't improve during training. 

To fix this, 

```
prompt = generate_prompt(R1_ZERO_PROMPT, data["problem"])
label = generate_label("{solution} </think> <answer> {answer} </answer>", data["solution"], data["answer"])
```

The model starts learning to generate text in the right format (and the reward improves) right away! A perfect demonstration that data is so critical for DL. 

### Hyperparameter Search

Hyperparameter search (batch size and learning rate). Search space: batch size in {8, 16, 32}, lr: {2e-4, 1e-4, 5e-5}. 

<img src="https://raw.githubusercontent.com/yyin-dev/image_cloud/main/Picsee/image-20250917123218980_545isL.jpeg" alt="image-20250917123218980" style="zoom:50%;" />

Batch size of 8, learning rate of 5e-5 works the best. 

To evaluate the performance on different dataset sizes, I did two flavors of experiments:

### Experiment 1: One epoch on different dataset sizes. 

Total number of examples = unique examples.

<img src="https://raw.githubusercontent.com/yyin-dev/image_cloud/main/Picsee/image-20250917124415879_bwqK0R.jpeg" alt="image-20250917124415879" style="zoom:50%;" />

It's clearly that larger dataset doesn't necessarily improve validation accuracy. In fact, one epoch on 128 is enough to get 22% validation accuracy, which is the best. 

### Experiment 2: The same total number of examples, different number of epochs. 

Make the model see the same number of examples, varying number of unique examples. 

<img src="https://raw.githubusercontent.com/yyin-dev/image_cloud/main/Picsee/image-20250917124808181_T4bG47.jpeg" alt="image-20250917124808181" style="zoom:50%;" />

This is result from batch size of 16, learning rate of 1e-4, but I believe the qualitative result is transferrable. We SFT on 1024 examples, each with 128, 256, 512, and 1024 unique examples. 

This clearly shows that it's better to run more epochs on a smaller dataset rather than fewer epoch on larger dataset.



The MATH dataset we obatined is probably already filtered (see `filter_dataset.py`). Less than 1% of training dataset is bad so I don't think this will have a meaningful impact on the validation accuracy. The private dataset provided for the course is probably not pre-filtered so this filtering would likely produce meaningful improvements. 

## Problem (expert_iteration_experiment)

In normal SFT, the training dataset is a fixed, pre-collected dataset. In expert-iteration (EI), the dataset is generated by the model itself. EI is a iterative, self-improving process. 

Hyperparameters: 

* Expert iteration batch size 
* rollout / G
* SFT epochs
* Other SFT params: batch size, microbatch size, learning rate (already tuned in SFT experiment). 

### Curves

Varying epoch, same EI batch size and rollout. For EIbs=512, increasing epoch does improve accuracy significantly when G=4, but not when G=2. It's probably a good idea to use epoch>=4?

<img src="https://raw.githubusercontent.com/yyin-dev/image_cloud/main/Picsee/image-20250918090116170_fyPcHV.jpeg" alt="image-20250918090116170" style="zoom:50%;" />

<img src="https://raw.githubusercontent.com/yyin-dev/image_cloud/main/Picsee/image-20250918090024415_iPX1A0.jpeg" alt="image-20250918090024415" style="zoom:50%;" />

Varying rollout G, same EI batch size (512) and epoch (2, 4, 8): Increasing G improves accuracy.

<img src="https://raw.githubusercontent.com/yyin-dev/image_cloud/main/Picsee/image-20250918090420197_SmwtvS.jpeg" alt="image-20250918090420197" style="zoom:50%;" />

Varying EI batch size and G, fix (EI batch size * G) and epoch (2, 4, 8): (ei_batch_size=512, G=4) > (ei_batch_size=2048, G=1) > (ei_batch_size=1024, G=2). 

<img src="https://raw.githubusercontent.com/yyin-dev/image_cloud/main/Picsee/image-20250918220908902_VK4baa.jpeg" alt="image-20250918220908902" style="zoom:50%;" />

### Entropy

Per-token entropy for ei_batch_size=2048, which shows an increasing trend, meaning the model is becoming less confident. 

<img src="https://raw.githubusercontent.com/yyin-dev/image_cloud/main/Picsee/image-20250918221314817_RVrG1i.jpeg" alt="image-20250918221314817" style="zoom:50%;" />

### Comparison with vanilla SFT

With vanilla SFT, the best validation accuracy is 33% with: unique_examples=128, epochs=16, batch size=8, lr=5e-5.

With expert iteration, the best accuracy is 36% with: 512 ei_batch_size, G=4, epochs=8, batch size=8, lr=5e-5.

## Problem (grpo_learning_rate)

![image-20250921103824310](https://raw.githubusercontent.com/yyin-dev/image_cloud/main/Picsee/image-20250921103824310_BidF5C.jpeg)

The best learning rate is 3e-5, while 1e-4 is too large and diverges and 1e-6 is too small to do meaningful learning. 

Observations on other metrics: 

* The train reward is much more noisy than valiation reward. 

* The per-token entropy shows a slowly decreasing trend, meaning the model is becoming more confident. 
* For good learning rates (i.e. 1e-5 and 3e-5), the gradient norm almost 5x around Step 11. It's unclear what happened there?

## Problem (grpo_baselines)

![image-20250921105411670](https://raw.githubusercontent.com/yyin-dev/image_cloud/main/Picsee/image-20250921105411670_S3N2xf.jpeg)

The eval_reward_mean curve clearly shows that baseline significantly helped with training. 

Observations on other metrics:

* The per-token entropy doesn't show the downwards trend, which exists when using baseline.
* The gradient norm spike wasn't observed either. 

## Problem (think_about_length_normalization)

Assumes that the constant normalizer in `masked_normalize` is `max_gen_len`, `masked_mean` is less influenced by sequence length than `masked_normalize`. In `masked_normalize`, the loss (and thus gradients) for a sequence is basically scaled down by `seq_len / max_seq_len`. 

Arguments for `masked_mean`:

* When two sequences have the same advantage, the model updates more for the longer sequence, which feels wrong?

Arguments for `masked_normalize`:

* Maybe harder problems require more thinking and thus longer sequences, while eaiser problems require less thinking and thus shorter sequences, so it's right to optimize more for harder problems? 

## Problem  (grpo_length_normalization)

![image-20250922203724715](https://raw.githubusercontent.com/yyin-dev/image_cloud/main/Picsee/image-20250922203724715_gCTfeU.jpeg)

`Masked_mean` works much better than `masked_normalize`. The loss and gradient norm have smaller scale with `masked_mean`, which is probably better for learning (?).

## problem (grpo_group_standard_normalization)

![image-20250923001241748](https://raw.githubusercontent.com/yyin-dev/image_cloud/main/Picsee/image-20250923001241748_rtUPCx.jpeg)

Yellow: without group std normalization. 

Green: with group std normazliation.

Without std normalization performs better. It seems that the loss and gradient norm is on average lower when w/o std normalization (which makes sense b/c the group norm is a number smaller than 1, and dividing by it scales up the loss and gradient).

## Note: learning rate scheduling

GRPO involves an outer loop and an inner loop. Conceptually it's like (consider on-policy)

```
for grpo_step in range(num_grpo_steps):
  prompts, ground_truths = sample(train_dataset)
  rollouts = model.generate(prompts)
  rewards = rewards(rollouts, ground_truths)
  old_log_probs = get_log_probs(model, prompts)
  
  for step in range(n_train_steps_per_rollout_batch):
    policy_log_probs = get_log_probs(model, prompts)
    loss = policy_gradient_loss(policy_log_probs, old_log_probs, rewards, ...)
    # gradient accumulation
    # optimizer step
```

Previously I was applying lr scheduling to the inner loop. And previously we were using `n_train_steps_per_rollout_batch = 1` because `rollout_batch_size = train_batch_size` and `n_epocsh_per_rollout_batch = 1`, so effectively we are using constant learning rate throughout the process.

We updated to apply lr scheduling across the every optimizer steps in this commit: https://github.com/yyin-dev/cs336-a5/commit/26e72fb1699a046a2bc2934d7be4bf817a0eba59

However, it seems that the model learns better with a large lr from the very beginning.

![image-20250924133456426](https://raw.githubusercontent.com/yyin-dev/image_cloud/main/Picsee/image-20250924133456426_Pb8Xz4.jpeg)

This is especially puzzling given that we are doing a very small number of warmup steps, and in the run with lr-scheduling, the lr is very close to the peak lr...

## Note: attempt to use `torch.compile`

I tried using `torch.compile` to improve performance. The code runs but I observed significantly worse training results.

* without torch.compile: https://wandb.ai/yueyin-dev-weights-biases/grpo-experiment/runs/pvkweazs/overview
* with torch.compile: https://wandb.ai/yueyin-dev-weights-biases/grpo-experiment/runs/mr9r1h8b/overview

I Googled and discussed with ChatGPT, and it seems that `torch.compile` might cause numerical divergence because:

* it might introduce numerical instability when fusing ops
* if the model contains conditionals or dynamic shapes, `torch.compile` silents inserts graph breaks, which might be numerically different

![image-20250924001805517](https://raw.githubusercontent.com/yyin-dev/image_cloud/main/Picsee/image-20250924001805517_qqjIbU.jpeg)

I also tried `mode="reduce-overhead"`, it fails with error message: RuntimeError: Inplace update to inference tensor outside InferenceMode is not allowed.You can make a clone to get a normal tensor before doing inplace update.See https://github.com/pytorch/rfcs/pull/17 for more details.

I think this means an **inference tensor** (i.e., a tensor created under `torch.inference_mode()` or equivalent internal path) is being modified **in-place**. I don't chase this down...

## Note: inconsistent/non-deterministic runs

The two runs are run with the same hyperparameters on different git revisions, but with very different results. 

https://wandb.ai/yueyin-dev-weights-biases/grpo-experiment/runs/5kt9uk4x/overview

https://wandb.ai/yueyin-dev-weights-biases/grpo-experiment/runs/pfmmqzua/overview

The later run (yellow) performs much better than an earlier run (blue). However, the git commits in between look benign and shouldn't introduce meaningful changes to the training process. 

![image-20250923111122948](https://raw.githubusercontent.com/yyin-dev/image_cloud/main/Picsee/image-20250923111122948_cPOcqi.jpeg)

I used `git bisect` to try to find the commit that caused the change:

* https://wandb.ai/yueyin-dev-weights-biases/grpo-experiment/runs/x05uwr4l/overview
* https://wandb.ai/yueyin-dev-weights-biases/grpo-experiment/runs/1032umce/overview

![image-20250923111403635](https://raw.githubusercontent.com/yyin-dev/image_cloud/main/Picsee/image-20250923111403635_HW3u0n.jpeg)

The two bisect runs all look closer to the later (yellow) run, which points me to the commit that's just a documentation change... Here are the commands I ran

```
$ git bisect start
$ git bisect bad 60323cccef6065988258d8326c34cc3aa0035ba5
$ git bisect good cb29ee762557d7d4618108376632d4639c8a7777
Bisecting: 3 revisions left to test after this (roughly 2 steps)
[baa4edb4fb68a7b727c2dee04a10d5d3d5a739b7] Refactor metrics logging

$ git bisect bad
Bisecting: 0 revisions left to test after this (roughly 1 step)
[576f24c1310689f1db5623a6cc3763e5d4c6e3e3] On-policy loss type comparison

$ git bisect bad
Bisecting: 0 revisions left to test after this (roughly 0 steps)
[3f86b34ed08c7cff0e28dd2a6ddc455aa74ee9d2] LR tuning writeup
```

I discussed this with Claude and did some investigation. We considered the following reasons:

* I didn't set `seed` in the `SamplingParams` for evaluation, so that's why we observed different in evaluation reward. However, this doesn't explain the training difference - in out GRPO setup, evaluation doesn't affect training at all.
* There are some vLLM non-determinism even though I'm using `seed` in rollout `SamplingParams`, which caused rollouts to be non-deterministic. We printed out sampled rollout at the first step and it looks the same across runs. However, this doesn't guarantee that vLLM is deterministic when `seed` is set. 
  * vLLM non-determinism even with a single GPU and static dataset
    * Parallel GPU operations: GPU is massively parallel and floating-point math is not associative, meaing `(a+b)+c` is not always equal to `a + (b+c)` due to floating point imprecision. 
    * Compounding errors: due to the autoregressive nature of LLM inference, small difference can compound. Update: this is probably the culprit! https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
    * Even with static dataset, vLLM's core optimization is dynamic batching which is not deterministic. 
    * To maximize determinism, need to set a bunch of other environment variables and flags like `torch.use_deterministic_algorithms(True)`, run with `temperature=0` and `top_p=1.0`, etc. 
* It's possible that the first run (in blue) was an outlier, and later runs are reasonably consistent...

## problem (grpo_off_policy, grpo_off_policy_sweep)

Results for epoch=1, train_batch_size=64,128,256

![image-20250924134118450](https://raw.githubusercontent.com/yyin-dev/image_cloud/main/Picsee/image-20250924134118450_eb4VVI.jpeg)

## I stopped here...

The non-determinism and training instability is really frustrating. Runs with the exact same hyperparameters produce very different results. For example, lr=3e-5, epoch=2, train_batch_size=256: The difference is particularly large in train_rewards. Also, the training seems to have diverged, but it works totally fine when epoch=1.

![image-20250924134153040](https://raw.githubusercontent.com/yyin-dev/image_cloud/main/Picsee/image-20250924134153040_e6SzY9.jpeg)

I guess I really learned that (1) RL training is unstable and (2) LLMs are non-deterministic (which makes it painful to get reliable/conclusive result...).
