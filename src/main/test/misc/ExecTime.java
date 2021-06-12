package main.test.misc;

import java.io.File;
import java.util.Arrays;

import main.Layer;
import main.Network;
import main.math.ActivationFunction;
import main.math.CostFunction;
import main.math.Initializer;
import main.test.mnist.DigitDataSet;
import main.test.mnist.TestMNIST;
import main.training.DataSet;
import main.training.Match;

/**
 * Small example showing (approximately) the execution time for a feedforward phase
 */
public class ExecTime {
	
	public static void main (String ... strings) {
		
		Layer input_to_hidden = new Layer(784, 16, ActivationFunction.Sigmoid, Initializer.XavierNormal);
		Layer hidden_to_hidden = new Layer(16, 16, ActivationFunction.Sigmoid, Initializer.XavierNormal);
		Layer hidden_to_output = new Layer(16, 10, ActivationFunction.Sigmoid, Initializer.XavierNormal);

		float learningRate = 0.05f;

		Network network = new Network.Builder(input_to_hidden).addLayers(hidden_to_hidden, hidden_to_output)
				.setLearningRate(learningRate).setCostFunction(CostFunction.HalfQuadratic).compile();

		int testNumber = 1;
		
		File dataFile = new File(TestMNIST.class.getResource("/resources/mnist/train-images.idx3-ubyte").getPath());
		File labelFile = new File(TestMNIST.class.getResource("/resources/mnist/train-labels.idx1-ubyte").getPath());
		DataSet dataSet = new DigitDataSet().createSet(dataFile, labelFile, testNumber);	
				
		double start = System.currentTimeMillis();
		
		for (Match match : dataSet) { // only one match

			// it does not change whether the net is trained or not
			float [] output = network.feedforward(match.getInput());
			System.out.println(Arrays.toString(output));
			
		}
		
		double end = System.currentTimeMillis();
		
		double avg = (end - start) / (double)testNumber;
		
		System.out.println(String.format("Elapsed time: %f ms", avg));
	}
}
