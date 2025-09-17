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