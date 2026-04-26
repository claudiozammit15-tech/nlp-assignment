prompt_templates={
    "r0_input_template": '''### Instruction
You are a helpful assistant aiming to answer the following question with reasoning.
Please output in the following format:
```
### Answer
<your step-by-step reasoning>
Final answer: <your final answer>
```
%sPlease keep your output concise and limit it to 500 words. Use the most short reasoning to get the correct answer. Do not output unrelated content!

### Question
%s
''',
    "test_input_template": '''### Instruction
You are a helpful assistant aiming to answer the following question with step-by-step reasoning.
Please output in the following format:
```
### Answer
<your step-by-step reasoning>
Final answer: <your final answer>
```
%sDo not output unrelated content!

### Question
%s
''',
    "test_verify_input_template": '''Please refine your previous answer to correct potential errors.
Please output in the following format:
```
### Answer
<your refinement>
Final answer: <your updated answer>
```
%s
''',
    "test_feedback_input_template": '''### Instruction
You are a helpful assistant aiming to provide a feedback to other agents.
Please check another agent's answer and correct the potential incorrect knowledge, faulty reasoning or format errors in its answer.
Please output in the following format:
```
### Answer
<your feedback>
Final answer: <your updated answer>
```
%s

### Question
%s

### Answer from Another Agent
%s
''',
    "train_input_template": '''### Instruction
You are a helpful assistant aiming to answer the following question with step-by-step reasoning.
Please output in the following format:
```
### Answer
<your step-by-step reasoning>
Final answer: <your final answer>
```
%sDo not output unrelated content!

### Question
%s
''',
    "self_reflection_template": '''### Instruction
You are a helpful assistant aiming to provide a feedback to yourself.
Unfortunately, your previous answer to the question was incorrect. Given the correct final answer, please reflect on it and provide feedback on your incorrect response. You need to identify the potential incorrect knowledge, faulty reasoning or format errors in your incorrect response. Please check if your incorrect final answer has any formatting errors compared to the correct answer.
Please output in the following format:
```
### Feedback
<your feedback>
```
You don't need to apologize.
Only output your feedback. Please keep your output concise and limit it to 500 words. Provide your most concise feedback. Do not output a new answer.

### Question
%s

### Correct Final Answer
%s

### Your Incorrect Answer
%s
Fianl answer: %s
''',
    "self_reflection_test_template": '''### Instruction
You are a helpful assistant aiming to provide a feedback to yourself.
Please check the correctness of your answer, reflect on it, and provide feedback on your potenial incorrect response. You need to identify the potential incorrect knowledge, faulty reasoning or format errors in your incorrect response.
Please output in the following format:
```
### Feedback
<your feedback>
```
You don't need to apologize.
Only output your feedback. Please keep your output concise and limit it to 500 words. Provide your most concise feedback. Do not output a new answer.

### Question
%s

### Your Answer
%s
Fianl answer: %s
''',
    "critic_template": '''### Instruction
You are a helpful assistant aiming to provide a feedback to other agents.
Congratulation! You answered the question correctly. Based on your correct answer, please help another agent who answered incorrectly. You should provide feedback by pointing out the potential incorrect knowledge, faulty reasoning or format errors in its response. Please check if its incorrect final answer has any formatting errors compared to the correct answer.
Please output in the following format:
```
### Feedback
<your feedback>
```
Only output your feedback. Please keep your output concise and limit it to 500 words. Provide your most concise feedback. Do not output a new answer.

### Question
%s

### Your Correct Answer
%s
Final answer: %s

### Incorrect Answer from Another Agent
%s
Final Answer: %s
''',
    "critic_test_template": '''### Instruction
You are a helpful assistant aiming to provide a feedback to other agents.
Based on your answer, please provide feedback to another agent. You should provide feedback by pointing out the potential incorrect knowledge, faulty reasoning or format errors in its response.
Please output in the following format:
```
### Feedback
<your feedback>
```
Only output your feedback. Please keep your output concise and limit it to 500 words. Provide your most concise feedback. Do not output a new answer.

### Question
%s

### Your Answer
%s
Final answer: %s

### Answer from Another Agent
%s
Final Answer: %s
''',
    "debate_input_template": '''### Instruction
You are a helpful assistant aiming to answer the following question with reasoning.
You are debating the input question with other agents. You are provided with all answers from the last round, along with their self-reflections and the feedback received. Please carefully consider these answers and output a new answer.
Please output in the following format:
```
### Answer
<your step-by-step reasoning>
Final answer: <your final answer>
```
%sPlease keep your output concise and limit it to 500 words. Use the most short reasoning to get the correct answer. Do not output unrelated content! Do not explicitly mention other agents; instead, internally consider their answers.

### Question
%s

### Last Round Answers
%s
''',
    "train_debate_input_template": '''### Instruction
You are a helpful assistant aiming to answer the following question with step-by-step reasoning.
You are debating the input question with other agents. You are provided with all answers from the last round, along with their self-reflections and the feedback received. Please carefully consider these answers and output a new answer.
Please output in the following format:
```
### Answer
<your step-by-step reasoning>
Final answer: <your final answer>
```
%sDo not output unrelated content!

### Question
%s

### Last Round Answers
%s
'''
}

choice_prompts={
    "SQA": "Only choose yes or no as your final answer. ",
    "GSM8K": "Only use a single numeric value as your final answer. ",
    "ECQA": "Only choose 1,2,3,4,5 representing your choice as your final answer. ",
    "ARC": "Only choose A,B,C,D representing your choice as your final answer. ",
    "MATH": "Only use a mathematical expression as your final answer and wrap the final answer with $. Here are some examples without reasoning.\nProblem: What is $\left(\\frac{7}{8}\\right)^3 \cdot \left(\\frac{7}{8}\\right)^{-3}$?\nFinal answer: $1$\nProblem: Find the distance between the points $(2,1,-4)$ and $(5,8,-3).$\nFinal answer: $\sqrt{59}$\nProblem: The faces of an octahedral die are labeled with digits $1$ through $8$. What is the probability, expressed as a common fraction, of rolling a sum of $15$ with a pair of such octahedral dice?\nFinal answer: $\\frac{1}{32}$\nProblem: Solve \[x + 2 = 5 \]\nFinal answer: $3$\n\nRemember to output reasoning steps in your answers. ",
    "MATH_train": "Only use a mathematical expression as your final answer and wrap the final answer with $.",
    "MATH500": "Only use a mathematical expression as your final answer and wrap the final answer with $. Here are some examples without reasoning.\nProblem: What is $\left(\\frac{7}{8}\\right)^3 \cdot \left(\\frac{7}{8}\\right)^{-3}$?\nFinal answer: $1$\nProblem: Find the distance between the points $(2,1,-4)$ and $(5,8,-3).$\nFinal answer: $\sqrt{59}$\nProblem: The faces of an octahedral die are labeled with digits $1$ through $8$. What is the probability, expressed as a common fraction, of rolling a sum of $15$ with a pair of such octahedral dice?\nFinal answer: $\\frac{1}{32}$\nProblem: Solve \[x + 2 = 5 \]\nFinal answer: $3$\n\nRemember to output reasoning steps in your answers. ",
    "MMLUProPhys": "Only choose A,B,C,D,E,F,G,H,I,J representing your choice as your final answer. ",
    "MMLUProChem": "Only choose A,B,C,D,E,F,G,H,I,J representing your choice as your final answer. ",
    "MMLUProMath": "Only choose A,B,C,D,E,F,G,H,I,J representing your choice as your final answer. ",
    "MMLUProBiol": "Only choose A,B,C,D,E,F,G,H,I,J representing your choice as your final answer. ",
    "MMLUProComp": "Only choose A,B,C,D,E,F,G,H,I,J representing your choice as your final answer. "
}
