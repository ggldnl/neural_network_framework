package main.test.emnist;

import java.io.File;
import java.io.IOException;

import javax.swing.JFileChooser;
import javax.swing.UIManager;
import javax.swing.filechooser.FileNameExtensionFilter;

import main.Layer;
import main.Network;
import main.math.ActivationFunction;
import main.math.CostFunction;
import main.math.Initializer;
import main.training.DataSet;
import main.training.Match;

/**
 * Self contained example to show the EMNIST dataset use case.
 */
public class TestEMNIST {

	/*
	 * Efficiency over 20800 predictions, 100 epochs & batch size = 5: 83.00%
	 */

	private static File selectFile(JFileChooser chooser, String dialogTitle) {

		chooser.setDialogTitle(dialogTitle);
		chooser.resetChoosableFileFilters();
		
		int n = chooser.showOpenDialog(null);
		if (n == JFileChooser.APPROVE_OPTION)
			return chooser.getSelectedFile();

		throw new RuntimeException ("Open command cancelled by user.");
	}
	
	private static File saveOnFile(JFileChooser chooser, String dialogTitle) {
		return saveOnFile(chooser, System.getProperty("user.home") + "/Desktop/", dialogTitle);
	}
	
	private static File saveOnFile(JFileChooser chooser, String workingDirectory, String dialogTitle) {
		
		try {

			chooser.setDialogTitle(dialogTitle);
			chooser.setSelectedFile(new File("config." + Network.ext)); // default filename
			chooser.setFileFilter(new FileNameExtensionFilter("network file", Network.ext)); // filter extension
			
			int n = chooser.showSaveDialog(null);
			if (n == JFileChooser.APPROVE_OPTION) {
				String filename = chooser.getSelectedFile().toString();
			    if (!filename.endsWith(Network.ext))
			        filename += '.' + Network.ext;
			    
			    File res = new File (filename);
			    res.createNewFile();
			    
				return res;
			}
			
		} catch (IOException ioe) {
			ioe.printStackTrace();
		}
			
		throw new RuntimeException("Open command cancelled by user.");
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

	public static void main(String... strings) {

		// set look and feel for the JFileChooser
		try {
			UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		JFileChooser chooser = new JFileChooser();

		
		Layer input_to_hidden1 = new Layer(784, 32, ActivationFunction.Leaky_ReLU, Initializer.XavierNormal);
		Layer hidden1_to_hidden2 = new Layer(32, 32, ActivationFunction.Leaky_ReLU, Initializer.XavierNormal);
		Layer hidden2_to_output = new Layer(32, 26, ActivationFunction.Sigmoid, Initializer.XavierNormal);

		float learningRate = 0.05f;

		Network network = new Network.Builder(input_to_hidden1).addLayers(hidden1_to_hidden2, hidden2_to_output)
				.setLearningRate(learningRate).setCostFunction(CostFunction.Quadratic).compile();

		// training
		File dataFile = selectFile(chooser, "Select training binary data file");
		File labelFile = selectFile(chooser, "Select training label file");
		DataSet dataSet = new LetterDataSet().createSet(dataFile, labelFile);

		int batchSize = 5;
		int epochs = 100;

		network.train(dataSet, batchSize, epochs);

		dataSet = null; // prevent GC overhead

		// saving configuration
		try {
			System.out.println("Trying to save");
			File config = saveOnFile(chooser, "Save network configuration");
			network.save(config);
		} catch (Exception e) {
			e.printStackTrace();
		}

		// testing
		dataFile = selectFile(chooser, "Select testing binary data file");
		labelFile = selectFile(chooser, "Select testing label file");
		DataSet testSet = new LetterDataSet().createSet(dataFile, labelFile);

		testSet.shuffle();
		
		try {

			int correctPredictions = 0;
			int totalPredictions = 0;

			for (Match match : testSet) {

				float[] output = network.feedforward(match.getInput());
				int label = match.getLabel();
				int guess = prediction(output);

				if (totalPredictions < 10) { // show the process
					System.out.printf(
							"Label: %d\tChar conversion: %c\tGuess: %d\tChar conversion: %c\tPredicted: %s\t\n", label,
							(char) (label + 97), guess, (char) (guess + 97), label == guess);
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
}
