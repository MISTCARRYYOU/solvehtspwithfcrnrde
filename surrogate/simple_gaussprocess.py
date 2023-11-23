import gpytorch
from surrogate.Condigure import *

# Training data is 100 points in [0,1] inclusive regularly spaced
train_dataset = FitDataset_train()

train_x = train_dataset.x_data
train_y = train_dataset.y_data
# train_x = torch.randn(1500, 4800)
# train_x = torch.linspace(0, 1, 100)
# True function is sin(2*pi*x) with Gaussian noise
# train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * math.sqrt(0.04)
# train_y = torch.randn(1500)
# We will use the simplest form of GP model, exact inference


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x, train_y, likelihood)

test_dataset = FitDataset_test()
loss_func = torch.nn.MSELoss()
model.eval()
test_x = test_dataset.x_data
test_y = test_dataset.y_data

f_preds = model(test_x)
y_preds = likelihood(model(test_x))

f_mean = f_preds.mean
f_var = f_preds.variance
f_covar = f_preds.covariance_matrix
f_samples = f_preds.sample()

print(f_samples, '\n', y_preds)
