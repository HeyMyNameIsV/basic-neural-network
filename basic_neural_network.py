import numpy as np
import pandas as pd
import functions 

data, weights = functions.generate_values(50, 3)

learn_rate = 0.1
bias = 0.01
epoch = 100
epoch_loss = []

def train(data, weights, epoch, learn_rate, bias):
    global epoch_loss
    for e in range(epoch):
        individual_loss = []
        for i in range(len(data)):
            feature = data.loc[i][:-1]  
            target = data.loc[i][-1]    
            w_sum = functions.get_weight_sum(feature, weights, bias)
            prediction = functions.sigmoid_function(w_sum)
            loss = functions.cross_entropy(target, prediction)
            individual_loss.append(loss)
            weights, bias = functions.gradient_descent(feature, target, weights, bias, learn_rate, prediction)
        
        average_loss = sum(individual_loss) / len(individual_loss)
        epoch_loss.append(average_loss)
        print(f"Epoch {e+1}/{epoch}, Loss: {average_loss}")


train(data, weights, epoch, learn_rate, bias)


df = pd.DataFrame(epoch_loss, columns=['Average Loss'])


df_plot = df.plot(kind="line", grid=True, title="Training Loss Over Epochs").get_figure()


df_plot.savefig("Training_Loss.pdf")
