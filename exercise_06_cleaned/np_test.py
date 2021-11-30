import numpy as np

# y_out = np.array([[1, 2, 3],
#                   [4, 5, 6],
#                   [7, 8, 9],
#                   [5, 6, 7]])
# y_truth = np.array([1, 2, 0, 1])
#
# N, C = y_out.shape
# y_truth_one_hot = np.zeros_like(y_out)
# y_truth_one_hot[np.arange(N), y_truth] = 1
#
# # Transform the logits into a distribution using softmax.
# aaa = np.max(y_out, axis=1, keepdims=True)
# bbb = y_out - np.max(y_out, axis=1, keepdims=True)
# y_out_exp = np.exp(y_out - np.max(y_out, axis=1, keepdims=True))
# y_out_probs = y_out_exp / np.sum(y_out_exp, axis=1, keepdims=True)

# Compute the loss for each element in the batch.
# loss = -y_truth_one_hot * np.log(y_out_probs)
# loss = loss.sum(axis=1).mean()
# print(loss)

# self.cache['probs'] = y_out_probs

y_out_probs = np.array([[0.3, 0.7],
                        [0.8, 0.2],
                        [0.1, 0.9]])
y_truth = np.array([1, 0, 1])

N, C = y_out_probs.shape
gradient = y_out_probs
gradient[np.arange(N), y_truth] -= 1
gradient /= N
print(gradient)

