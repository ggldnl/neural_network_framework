package main.test.misc;

import java.io.File;
import java.io.IOException;

import javax.swing.JFileChooser;
import javax.swing.UIManager;
import javax.swing.filechooser.FileFilter;

import main.Network;
import main.test.mnist.DigitDataSet;
import main.training.DataSet;
import main.training.Match;

/**
 * Self contained example to show how to restore a serialized network.
 */
public class RestoreConfig {

	private static class RestoreFileFilter extends FileFilter {

		public boolean accept(File file) {
			if (file.isDirectory())
				return true;
			String fname = file.getName().toLowerCase();
			return fname.endsWith(Network.ext);
		}

		public String getDescription() {
			return "Serializable Network File";
		}
	}

	public static void main(String... strings) throws IOException {

		try {
			UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
		} catch (Exception e) {
			e.printStackTrace();
		}

		JFileChooser fileChooser = new JFileChooser();
		fileChooser.setFileFilter(new RestoreFileFilter());
		int n = fileChooser.showOpenDialog(null);
		if (n == JFileChooser.APPROVE_OPTION) {

			// restore the network
			File restore = fileChooser.getSelectedFile();
			Network network = Network.restoreNetwork(restore);

			/*
			 * // create the dataset (EMNIST)
			 * 
			 * File dataFile = new File(RestoreConfigurationExample.class
			 * .getResource("/resources/emnist/emnist-letters-test-images-idx3-ubyte").
			 * getPath());
			 * 
			 * File labelFile = new File(RestoreConfigurationExample.class
			 * .getResource("/resources/emnist/emnist-letters-test-labels-idx1-ubyte").
			 * getPath());
			 * 
			 * DataSet set = new LetterDataSet().createSet(dataFile, labelFile);
			 */

			// create the dataset (MNIST)
			File dataFile = new File(
					RestoreConfig.class.getResource("/resources/mnist/t10k-images.idx3-ubyte").getPath());

			/*
			 * We can do without labels.
			 * 
			 * File labelFile = new File( RestoreConfigurationExample.class.getResource(
			 * "/resources/mnist/t10k-labels.idx1-ubyte").getPath());
			 */

			DataSet set = new DigitDataSet().createSet(dataFile);
			
			// shuffle the dataset
			set.shuffle();
			
			// testing (MNIST)
			try {

				int totalPredictions = 0;

				for (Match match : set) {
					
					float[] output = network.feedforward(match.getInput());
					int guess = prediction(output);

					if (totalPredictions < 10) {
						System.out.printf("Guess: %d\n", guess);
						System.out.println(match);
						System.out.println();

						Thread.sleep(1000); // little delay to observe the result
					}

					totalPredictions++;
				}

				System.out.printf("Total predictions: %d\n", totalPredictions);
			} catch (Exception e) {
				e.printStackTrace();
			}
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
