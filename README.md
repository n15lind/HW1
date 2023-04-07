# Homework 1: Model Fitting
**Neil Lindgren**

For this assignment, I was tasked with fitting models to a given set of data and minimizing the least-square errors of each of these models.

## Sec. I. Introduction and Overview
I was given the following data set to work with for this assignment:
```
X_data=np.arange(0,31)
Y_data=np.array([30, 35, 33, 32, 34, 37, 39, 38, 36, 36, 37, 39, 42, 45, 45, 41,
40, 39, 42, 44, 47, 49, 50, 49, 46, 48, 50, 53, 55, 54, 53])
```
I first had to fit a sinusoidal model of the form *f(x) = A cos(Bx) + C x + D* and optimize the parameters A, B, C, and D to minimize the least-square error. I then generated loss landscapes by fixing 2 parameters at a time and sweeping through the other two. I created a loss landscape for each combination of the parameters.

After that, I used the first 20 data points listed above as training data for additional models. I fit a line, a parabola, and a 19th degree polynomial to the training data, then used the remaining data points as test data to see how effective each model was by calculating the least-square error.

Finally, I used the first 10 data points and the last 10 data points as training data for the same 3 models above, and used the middle 10 points as test data.

## Sec. II. Theoretical Background
I optimized the parameters of my models by using the Nelder-Mead method to minimize the least-square error of the models. The least-square error is defined as
$$E = \sqrt{\frac{1}{n}\sum_{j=1}^{n}(f(x_j) - y_j)^2}$$
Note that the Nelder-Mead method finds local minima for the least-square error, and it's highly sensitive to the initial "guess" values that you use when implementing it. The point of the first section of this assignment was to find the "best" local minima to fit the model to the data.

For more information on the Nelder-Mead method, see https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method

## Sec. III. Algorithm Implementation and Development 
### Sinusoid Fit
First, I fit a sinusoidal model of the form *f(x) = A cos(Bx) + C x + D* to the data and calculated the least-squares error of it with the following code:
```
def lserror (c, x, y):
    return np.sqrt(np.sum((c[0]*np.cos(c[1]*x)+c[2]*x+c[3]-y)**2)/32)
```
I then set an array c0 with my initial guesses for each parameter:
```
c0 = np.array([20, np.pi/6, 1, 31])
```
I then implemented the Nelder-Mead method to minimize the least-square error:
```
res = opt.minimize(lserror, c0, args=(X_data, Y_data), method='Nelder-Mead')
```
Finally, I took the output values (the optimized parameters A, B, C, and D) from this optimization and put them in a sinusoidal model to see how well it fit the data.

### Loss Landscapes
For my loss landscapes, I fixed 2 variables at a time and swept the remaining two, then plotted the results on a heatmap to show where the least-square error reached local minima. I did this for every combination of the parameters. First, I defined the sweep ranges for each parameter:
```
A_range = np.linspace(0, 25, 101)
B_range = np.linspace(0, 1, 101)
C_range = np.linspace(0, 5, 101)
D_range = np.linspace(25, 35, 101)
```
I fixed two parameters, then created a meshgrid from the values of the remaining unfixed parameters. After that, I filled the meshgrid with the values of the loss landscape and plotted the results. Here is an example from when I fixed A and B:
```
# Create a meshgrid of parameter values
Xab, Yab = np.meshgrid(C_range, D_range)

# Initialize the loss landscape
Zab = np.zeros_like(Xab)

# Fix A and B parameters
A = c0[0]
B = c0[1]

# Sweep through C and D parameters
for i in range(Xab.shape[0]):
    for j in range(Xab.shape[1]):
        C = Xab[i,j]
        D = Yab[i,j]
        c = np.array([A, B, C, D])
        Zab[i,j] = lserror(c, X_data, Y_data)
```
I repeated this process for A & C fixed, A & D fixed, B & C fixed, B & D fixed, and finally C & D fixed.

### Using 20 Points as Training Data
For this part, I optimized my models using np.poly1d to come up with model parameters based on the training data. I then plotted the models and calculated their least-square errors to compare how they fit the data. The process was essentially the same, regardless of which points I used as training data, but the choice of training data yielded different results for the error of the models. Here is an example of fitting my models to the training data:
```
# Split the data into training and test data
X_train = X_data[:20]
Y_train = Y_data[:20]
X_test = X_data[20:]
Y_test = Y_data[20:]

# Fit a line to the training data
line_coeffs = np.polyfit(X_train, Y_train, 1)
line_fit = np.poly1d(line_coeffs)

# Compute the least square error for the line fit
line_lserror_train = np.sqrt(np.sum((line_fit(X_train) - Y_train)**2)/20)
line_lserror_test = np.sqrt(np.sum((line_fit(X_test) - Y_test)**2)/10)

# Fit a parabola to the training data
parabola_coeffs = np.polyfit(X_train, Y_train, 2)
parabola_fit = np.poly1d(parabola_coeffs)

# Compute the least square error for the parabola fit
parabola_lserror_train = np.sqrt(np.sum((parabola_fit(X_train) - Y_train)**2)/20)
parabola_lserror_test = np.sqrt(np.sum((parabola_fit(X_test) - Y_test)**2)/10)

# Fit a 19th degree polynomial to the training data
poly_coeffs = np.polyfit(X_train, Y_train, 19)
poly_fit = np.poly1d(poly_coeffs)

# Compute the least square error for the 19th degree polynomial fit
poly_lserror_train = np.sqrt(np.sum((poly_fit(X_train) - Y_train)**2)/20)
poly_lserror_test = np.sqrt(np.sum((poly_fit(X_test) - Y_test)**2)/10)
```

## Sec. IV. Computational Results

### Sinusoid Fit

The optimized parameters I found with the Nelder-Mead method are A=2.17177157, B=0.90932511, C=0.73248788, and D=31.45276293, resulting in the following plot:

![Sinusoid Fit](https://user-images.githubusercontent.com/130141391/230597214-77dc484f-0d66-4a58-815c-45c9cd72444a.png)

### Loss Landscapes

It's difficult to say exactly how many local minima exist for each combination of parameters. Some had one clear minimum, others had general regions that appeared to function as minima. It would be nearly impossible to count every single unique combination of parameters that made a local minima, so instead I will be counting each "minima region". I counted 2 for fixed A & B, 7 for fixed A & C, 9 for fixed A & D, 2 for fixed B & C, 2 for fixed B & D, and 8 for fixed C & D. Here are the loss landscapes, each one individually labelled with which parameters are fixed:

![ab](https://user-images.githubusercontent.com/130141391/230598165-96978287-6f7b-43ff-ab3d-a5ff7f4929eb.png)
![ac](https://user-images.githubusercontent.com/130141391/230598170-4e4d1b8e-b99e-4236-9940-6b188e7075bc.png)
![ad](https://user-images.githubusercontent.com/130141391/230598176-301d577e-75da-4287-a9bd-c76260c531d3.png)
![bc](https://user-images.githubusercontent.com/130141391/230598183-a6b37c7d-8e90-47ef-a4b4-846d6daaf149.png)
![bd](https://user-images.githubusercontent.com/130141391/230598192-cb1ea3bd-db2c-429c-82bf-c6022e12b5f7.png)
![cd](https://user-images.githubusercontent.com/130141391/230598201-720aba66-eea1-4fd6-aae7-7ae13f62fac1.png)

### Using 20 Points as Training Data
For the first case, where the first 20 points served as training data and the remaining points served as test data, the line fit had the highest training error but the lowest test error out of all of the models. The 19th degree polynomial had the best training error, but it had the worst test error by 10 orders of magnitude or so. The parabola fit had reasonable training error, but fairly bad test error. The plots for these 3 fits are shown below, with the train error and test error labelled on them:

![20 first](https://user-images.githubusercontent.com/130141391/230600669-be5b29d3-eca5-426f-8e6a-6a22f63ad021.png)

For the second case, where the first 10 points and the last 10 points served as training data and the remaining points served as test data, the parabola fit had the best test error. The training error was the same between the line fit and the parabola fit. Again, the 19th degree polynomial fit had the best training error but massive test error. The plots for these 3 fits are shown below, with the train error and test error labelled on them:

![10 and 10](https://user-images.githubusercontent.com/130141391/230601098-990dda60-6868-45bb-a045-252175d28551.png)

In both cases, it's clear that the 19th degree polynomial overfit the data, as it reached insane values outside of the training data.

## Sec. V. Summary and Conclusions

While models are powerful tools to fit data, it was easy for them to fall into local minima that were not useful. For the sinusiod model using the Nelder-Mead method, even small changes in my initial guesses could wildly change the resulting curve. The loss landscapes showed local minima for each parameter that could be useful, but I had to look at all of the loss landscapes at the same time to determine which of these minima were actually ideal. For the section using part of the data set as training data, the 19th degree polynomial overfit the data and provided bad results. The line model was the best when the first 20 points were the training data, as it continued the upward trend in the test data. The parabola was the best model when the first 10 points and the last 10 points were the training data, as its gentle curve matched the shape of the data better than the line did. Overall, I learned that I need to keep a close eye on any models I use to make sure that they don't try to incorrectly fit to the data.
