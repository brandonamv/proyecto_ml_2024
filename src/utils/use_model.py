import pickle


def search_trained_model():
  # loaded_model = pickle.load(open("src/models/model.pickle", 'rb'))
  # return loaded_model 
  return None

def get_model(model,exist_model):
  if(exist_model):
    return model
  else:
    exist_model = True
    model = search_trained_model()
    return model,exist_model
  
def predict(img,model):
  res = "something"
  return res
