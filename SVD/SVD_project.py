import numpy as np
import matplotlib.pyplot as plt

TrainMat = np.load('TrainDigits.npy')
TrainLab = np.load('TrainLabels.npy')

def get_training_digits(d, n=400):
    index = (TrainLab == d)
    A_d = TrainMat[:, index[0]][:, 0:n]
    return A_d

def get_svd(d):
    A_temp = get_training_digits(d)
    Ud_temp, Sd_temp, Vdt_temp = np.linalg.svd(A_temp, full_matrices=False)
    return Ud_temp, Sd_temp, Vdt_temp

def plot_singular_images (d: int): #d is the digit
    Ud_tmep, Sd_temp, Vdt_temp = get_svd(d)
    for i in range (0,3):
        U_i = Ud_tmep [:, i]
        U_i_reshaped = np.reshape(U_i, (28, 28)).T #reshapes into 28X28 
        plt.imshow(np.real(U_i_reshaped), cmap = 'gray')
        plt.title(f"Singular Image u_{i+1} for digit {d}")
        plt.show()

def plot_singular_values(d):
    Sd = get_svd(d)[1]
    plt.figure()
    plt.plot(Sd)
    plt.title(f'Singular Values for Digit {d}')
    plt.xlabel('Index')
    plt.ylabel('Singular Value')
    plt.grid(True)
    plt.show()  

""" print(plot_singular_values(3))
print(plot_singular_values(8))
 """
### To Do 2
TestMat = np.load('TestDigits.npy')
TestLab = np.load('TestLabels.npy')

max_k = 15
U_dict = {}

for d in range (10):
    U = get_svd(d)[0]
    U_dict[d] = U[:, :max_k]



k_values = range(5, 16)
accuracy_results = {d: [] for d in range(10)}

for k in k_values:
    residuals = np.zeros((10, TestMat.shape[1]))

    for d in range(10):
        U_k = U_dict[d][:, :k]

        ### Vectorized residual calculation for all 40 000 images
        projection = U_k @ (U_k.T @ TestMat)
        res_matrix = TestMat - projection

        ### 2Norm for each column
        residuals[d, :] = np.linalg.norm(res_matrix, axis=0, ord= 2)

    ### Find the predicted digit
    predictions = np.argmin(residuals, axis=0)

    ### Accuracy per digit
    test_labels = TestLab[0]
    for d in range(10):
        mask = test_labels == d
        acc = (predictions[mask]==d).mean() * 100
        accuracy_results[d].append(acc)
def plot_trendlines():
    plt.figure(figsize=(9, 6))
    for d in range(10):
        plt.plot(k_values, accuracy_results[d], label=f'Digit {d}')

        plt.xlabel('k (columns of U)')
        plt.ylabel('Percentage of success')
        plt.title('SVD-based Digit Classification Accuracy')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

#plot_trendlines()

###extra test:
def plot_only_5_for_low_k():
    max_k = 10
    k_values = range(1, max_k + 1)
    U_dict = {d: get_svd(d)[0][:, :max_k] for d in range(10)}

    test_labels = TestLab[0]
    digit_5_accuracy = []

    for k in k_values:
        residuals = np.zeros((10, TestMat.shape[1]))

        for d in range(10):
            U_k = U_dict[d][:, :k]
            projection = U_k @ (U_k.T @ TestMat)
            res_matrix = TestMat - projection
            residuals[d, :] = np.linalg.norm(res_matrix, axis=0, ord=2)

        predictions = np.argmin(residuals, axis=0)

        # Accuracy specifically for digit 5
        mask_5 = test_labels == 5
        acc_5 = (predictions[mask_5] == 5).mean() * 100
        digit_5_accuracy.append(acc_5)


    # Plot trend for Digit 5
    plt.figure(figsize=(7, 4.5))
    plt.plot(k_values, digit_5_accuracy, marker='o', color='purple', linewidth=2)
    plt.xlabel('k (columns of U)')
    plt.ylabel('Percentage of success (%)')
    plt.title('SVD Classification Accuracy - Digit 5 (k = 1 to 10)')
    plt.xticks(k_values)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_only_5_for_low_k()