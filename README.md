# GAwithPerceptrons
Investigating Genetic Algorithms for Training Perceptrons in Non-Linearly Separable Problems

we delve into the application of genetic algorithms specifically for training perceptrons—a fundamental unit of ANNs. We explore how genetic algorithms can be utilized to train perceptrons to solve non-linearly separable problems, such as the XOR (exclusive OR) problem. Our investigation aims to analyze the efficacy of genetic algorithms in evolving perceptrons to achieve high accuracy in classification tasks.

## Results

We conduct experiments using our implemented genetic algorithm and perceptron model. The experiments involve training perceptrons on both linearly separable and non-linearly separable problems. We track the evolution of the population over multiple generations and observe the best accuracy achieved by the evolved perceptron.

### Linearly Separable Problem:
In the case of linearly separable problems, such as logical OR and logical AND, our genetic algorithm consistently converges to a solution with 100% accuracy. This demonstrates the effectiveness of genetic algorithms in optimizing perceptrons for simple classification tasks.

### Non-Linearly Separable Problem (XOR):
When tackling the XOR problem, which is non-linearly separable, our genetic algorithm still manages to find a solution. However, the convergence to a high-accuracy solution is slower compared to linearly separable problems. Nevertheless, the evolved perceptron achieves a respectable accuracy, often around 75%, showcasing the adaptability of genetic algorithms even in challenging scenarios.

## 3.1 Experiments
Two experiments were conducted using the following parameters:
•	Experiment 1:
    o	Population size: 200
    o	Generations: 100
    o	Mutation rate: 0.7
    o	Data: Linearly separable (all data points are perfectly classifiable)
•	Experiment 2:
    o	Population size: 150
    o	Generations: 200
    o	Mutation rate: 0.8
    o	Data: Non-linearly separable (XOR problem)

### 3.2 Findings
•	Experiment 1:
    o	The GA achieved a perfect accuracy (1.0) on the linearly separable data set within 3 generations. This demonstrates the effectiveness of the GA in finding optimal weights and bias for a simple classification task.
 
  
  
  
•	Experiment 2:
    o	The GA achieved a maximum accuracy of 0.75 on the non-linearly separable XOR problem. This suggests that the GA may struggle with tasks that require more complex decision boundaries beyond the capabilities of a single perceptron.
  

