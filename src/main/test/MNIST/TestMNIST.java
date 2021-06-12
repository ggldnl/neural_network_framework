package main.test.mnist;

import java.io.File;

import main.Layer;
import main.Network;
import main.math.ActivationFunction;
import main.math.CostFunction;
import main.math.Initializer;
import main.training.DataSet;
import main.training.Match;

/**
 * Self contained example to show the MNIST dataset use case.
 */
public class TestMNIST {

	/*
	 * Efficiency over 10000 predictions, 100 epochs, batch size = 5, 2 hidden
	 * layers of 16 neurons each: 94.87%
	 * 
	 * Efficiency over 10000 predictions, 100 epochs, batch size = 5, 1 hidden layer
	 * of 16 neurons: 94.66%
	 */

	public static void main(String... strings) {

		Layer input_to_hidden = new Layer(784, 16, ActivationFunction.Sigmoid, Initializer.XavierNormal);
		Layer hidden_to_hidden = new Layer(16, 16, ActivationFunction.Sigmoid, Initializer.XavierNormal);
		Layer hidden_to_output = new Layer(16, 10, ActivationFunction.Sigmoid, Initializer.XavierNormal);

		float learningRate = 0.05f;

		Network network = new Network.Builder(input_to_hidden).addLayers(hidden_to_hidden, hidden_to_output)
				.setLearningRate(learningRate).setCostFunction(CostFunction.HalfQuadratic).compile();

		// training
		File dataFile = new File(TestMNIST.class.getResource("/resources/mnist/train-images.idx3-ubyte").getPath());
		File labelFile = new File(TestMNIST.class.getResource("/resources/mnist/train-labels.idx1-ubyte").getPath());
		DataSet dataSet = new DigitDataSet().createSet(dataFile, labelFile);

		network.verbose(false);

		int batchSize = 5;
		int epochs = 1;

		System.out.println(java.util.Arrays.toString(network.getLayers()[2].getBiases()));
		
		network.train(dataSet, batchSize, epochs);

		System.out.println(java.util.Arrays.toString(network.getLayers()[2].getBiases()));
		
		/*
		 * prevent GC overhead: when the old is no longer needed, it can be easily
		 * discarded.
		 */
		dataSet = null;

		/*
		 * Try to make the garbage collector run immediately. It may still not run, it
		 * depends on the Java Runtime Environment.
		 */
		// System.gc();

		// saving configuration
		try {
			String name = "config_digits";
			String outputpath = System.getProperty("user.home") + "/Desktop/" + name + '.' + Network.ext;
			File config = new File(outputpath);
			config.createNewFile();
			network.save(config);
		} catch (Exception e) {
			e.printStackTrace();
		}

		// testing
		dataFile = new File(TestMNIST.class.getResource("/resources/mnist/t10k-images.idx3-ubyte").getPath());
		labelFile = new File(TestMNIST.class.getResource("/resources/mnist/t10k-labels.idx1-ubyte").getPath());
		DataSet testSet = new DigitDataSet().createSet(dataFile, labelFile);

		try {

			int correctPredictions = 0;
			int totalPredictions = 0;

			for (Match match : testSet) {

				float[] output = network.feedforward(match.getInput());
				int label = match.getLabel();
				int guess = prediction(output);

				if (totalPredictions < 10) {
					System.out.printf("Label: %d\tGuess: %d\tPredicted: %s\t\n", label, guess, label == guess);
					System.out.println(match);
					System.out.println();

					Thread.sleep(1000); // little delay to observe the result
				}

				totalPredictions++;
				if (label == guess)
					correctPredictions++;
			}

			System.out.printf("Total predictions: %d\n", totalPredictions);
			System.out.printf("Correct predictions: %d\n", correctPredictions);
			System.out.printf("Accuracy: %f%%\n", correctPredictions * 100.0f / totalPredictions);
			
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private static int prediction(float[] output) {
		float max = -1;
		int index = -1;
		for (int i = 0; i < output.length; i++)
			if (output[i] > max) {
				max = output[i];
				index = i;
			}
		return index;
	}
}
