import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
import pandas as pd

### Handle data ###
file_path = r"C:\Users\lasse\Downloads\BerVet2_lokal\MiniProjects\Regression and least squares\LifeExpectancyData.csv"

data = pd.read_csv(file_path)

data.columns = data.columns.str.strip()

# Keep only the two columns
data = data[["Life expectancy", "Schooling"]]

data["Life expectancy"] = pd.to_numeric(
    data["Life expectancy"], errors="coerce"
)

data["Schooling"] = pd.to_numeric(
    data["Schooling"], errors="coerce"
)

cleaned_data = data[~np.isnan(data).any(axis=1)]

x = cleaned_data["Schooling"].to_numpy()
y = cleaned_data["Life expectancy"].to_numpy()


### Test cleaned data ###
def test_cleaned_data():
    tests_ok = 0

    if np.isnan(x).any() == False:
        tests_ok += 1

    if np.isnan(y).any() == False:
        tests_ok += 1

    if np.isfinite(x).all() == True:
        tests_ok += 1

    if np.isfinite(y).all() == True:
        tests_ok += 1

    if tests_ok == 4:
        return True
    else:
        return False


### Plot cleaned data ###
def plot_cleaned_data():

    plt.scatter(x, y, s=10)

    plt.title(
        "Life expectancy in relationship to schooling"
    )

    plt.xlabel("Years schooled")
    plt.ylabel("Life expectancy [years]")

    plt.legend(
        ["Years schooled vs Life expectancy"]
    )

    plt.show()
    plt.figure(1)


Pol_degree = 2


#new solution using vander
A = np.vander(x, Pol_degree +1, increasing = True )


### Normal equations ###
ATA = A.T @ A
ATy = A.T @ y

coefficients_normal = np.linalg.solve(
    ATA,
    ATy
)

### improved version without hardlock ###

def pol_normal(x):
    pol_norm = Polynomial(coefficients_normal)
    return pol_norm(x)

### QR decomposition ###
Q, R = np.linalg.qr(A)


from scipy.linalg import solve_triangular
coefficients_qr = solve_triangular(R, Q.T @ y)


### improved version without hardlock ###

def pol_qr(x):
    pol_qr = Polynomial(coefficients_qr)
    return pol_qr(x)


### Plot quadratic fit using normal equations ###
def plot_quad_normal():

    plt.scatter(
        x,
        y,
        s=10,
        label="Raw data"
    )

    years_schooled = np.linspace(np.amin(x), np.amax(x), 100)

    plt.plot(
        years_schooled,
        pol_normal(years_schooled),
        label="QuadraticNormal",
        color="green"
    )

    plt.title(
        "Life expectancy in relationship to schooling"
    )

    plt.xlabel("Years schooled")
    plt.ylabel("Life expectancy")

    plt.legend()
    plt.show()
    plt.figure(2)


### Plot quadratic fit using QR decomposition ###
def plot_quad_qr():

    plt.scatter(
        x,
        y,
        s=10,
        label="Raw data"
    )

    years_schooled = np.linspace(0, 20, 100)

    plt.plot(
        years_schooled,
        pol_qr(years_schooled),
        label="QuadraticQR",
        color="red"
    )

    plt.title(
        "Life expectancy in relationship to schooling"
    )

    plt.xlabel("Years schooled")
    plt.ylabel("Life expectancy")

    plt.legend()
    plt.show()
    plt.figure(3)

### Residual Analysis ###
def plot_residuals():
    y_predicted = pol_normal(x)
    residuals = y - y_predicted

    # Plot histogram of residuals
    plt.hist(residuals, bins=200)

    plt.title("Histogram of Residuals")
    plt.xlabel("Residual Value (y - y_predicted)")
    plt.ylabel("Frequency")

    plt.show()
    plt.figure(4)

### Condition Numbers ###
cond_A = np.linalg.cond(A)
cond_ATA = np.linalg.cond(ATA)

### Main function ###
def main():

    def _test_cleaned_data():

        if test_cleaned_data() == True:
            print("Cleaning of data ok.")
        else:
            print("Cleaning of data failed")

    _test_cleaned_data()

    print(
        "Visualize raw data by plotting cleaned data."
    )

    plot_cleaned_data()

    ### Comparing QR vs normal equations ###
    print(
        "Comparing the resulting quadratic functions "
        "from the normal equations and QR respectively."
    )

    print(
        f"The coefficients for the normal equation "
        f"approach are: {coefficients_normal}\n"
    )

    print(
        f"The coefficients for the QR approach are: "
        f"{coefficients_qr}\n"
    )

    print(
        "Visualize raw data alongside different "
        "quadratic fits."
    )

    plot_quad_normal()
    plot_quad_qr()
    plot_residuals()


    print(f"Condition number of matrix A (QR approach): {cond_A}")
    print(f"Condition number of matrix ATA (Normal equations): {cond_ATA}")

    def condition_number_calc():
        print('\n Verify that the condition numbers are correctly calculated. ')
        print(f'625599929.8927681/25011.995720560757 = {625599929.8927681/25011.995720560757}')
        print(f'25011.995720560757^2 = {25011.995720560757**2}')

    condition_number_calc()

    def precision_loss_estimate():
        ### Condition Numbers 
        cond_A = np.linalg.cond(A)
        cond_ATA = np.linalg.cond(ATA)

        # Precision loss estimatins
        digits_lost_qr = np.log10(cond_A)
        digits_remaining_qr = max(0, 16 - digits_lost_qr)

        digits_lost_normal = np.log10(cond_ATA)
        digits_remaining_normal = max(0, 16 - digits_lost_normal)

        # detials
        print("       NUMERICAL PRECISION & ACCURACY ANALYSIS       ")
        print(f"Polynomial Degree Used: {Pol_degree}")

        print("NORMAL EQUATIONS APPROACH (A^T * A):")
        print(f"  Coefficients:     {coefficients_normal}")
        print(f"  Condition Number: {cond_ATA:.4e}")
        print(f"  Digits Lost:      ~{digits_lost_normal:.2f} digits")
        print(f"  Digits Remaining: ~{digits_remaining_normal:.2f} digits out of 16")

        print("QR DECOMPOSITION APPROACH (Q * R):")
        print(f"  Coefficients:     {coefficients_qr}")
        print(f"  Condition Number: {cond_A:.4e}")
        print(f"  Digits Lost:      ~{digits_lost_qr:.2f} digits")
        print(f"  Digits Remaining: ~{digits_remaining_qr:.2f} digits out of 16")

        # Relative difference between the two coefficient solutions
        rel_diff = np.linalg.norm(coefficients_normal - coefficients_qr) / np.linalg.norm(coefficients_qr)
        print(f"Relative difference: {rel_diff}")

        # Compute absolute and relative errors
        abs_error = np.linalg.norm(coefficients_normal - coefficients_qr, ord=2)
        norm_qr = np.linalg.norm(coefficients_qr, ord=2)
        rel_error = abs_error / norm_qr


        print(f"Absolute Error ||x_NE - x_QR||: {abs_error:.6e}")
        print(f"Relative Error ||x_NE - x_QR|| / ||x_QR||: {rel_error:.6e}")

    precision_loss_estimate()


main()