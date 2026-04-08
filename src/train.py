from sklearn.linear_model import Ridge
def train_model(x, y):
    model = Ridge()
    model.fit(x, y)
    return model