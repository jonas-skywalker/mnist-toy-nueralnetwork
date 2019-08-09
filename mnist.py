import neuralnetwork

epochs = 5

training_data_file = open("/mnist/training/data/goes/here", "r")
training_data_list = training_data_file.readlines()
training_data_file.close()

nn = neuralnetwork.NeuralNetwork([784, 64, 10], 0.1)


print("Training...")
for e in range(epochs):
    print(e)
    count = 0
    for record in training_data_list:
        inputs = []
        # split the record by the ',' commas
        all_values = record.split(',')

        # scale and shift the inputs
        for i in range(1, len(all_values)):
            inputs.append((int(all_values[i]) / 255.0 * 0.99) + 0.01)

        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = [0.01 for x in range(10)]

        # all_values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99
        # go through all records in the training data set
        nn.linear_regression_gradient_descent(inputs, targets)
        if count % 1000 == 0:
            print(count)
        count += 1
        pass
    pass


# load the mnist test data CSV file into a list
test_data_file = open("/mnist/test/data/goes/here", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# test the neural network

# scorecard for how well the network performs, initially empty
scorecard = []

print("Testing...")
# go through all the records in the test data set
for record in test_data_list:
    inputs = []
    # split the record by the ',' commas
    all_values = record.split(',')

    # correct answer is first value
    correct_label = int(all_values[0])

    # scale and shift the inputs
    for i in range(1, len(all_values)):
        inputs.append((int(all_values[i]) / 255.0 * 0.99) + 0.01)

    # query the network
    outputs = nn.feed_forward(inputs)

    # the index of the highest value corresponds to the label
    label = neuralnetwork.matrix.Matrix.arg_max(outputs)

    # append correct or incorrect to list
    if label == correct_label:
        # network's answer matches correct answer, add 1 to scorecard
        scorecard.append(1)
    else:
        # network's answer doesn't match correct answer, add 0 to scorecard
        scorecard.append(0)
        pass

    pass

# calculate the performance score, the fraction of correct answers
scorecard_sum = sum(scorecard)
print("performance = " + str(scorecard_sum / len(scorecard)) + "%")

# nn.save_json("mnist_nn.json")
