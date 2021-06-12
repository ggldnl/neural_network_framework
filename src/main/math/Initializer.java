package main.math;

import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Initializer for weights and biases. Weights and biases should be chosen
 * carefully: choosing the parameters so that they are too weak would lead to
 * them to shutdown; choosing them so that they are too big would lead to an
 * uncontrolled amplification. 
 * The Initializer has the task of initializing the values of weights and biases. 
 * Various criteria can be adopted to do so, of which four have been implemented: 
 * XavierUniform, XavierNormal, Kaiming and Zero. 
 * Additional Initializers can be added. There are two methods to expose:
 * <i>initWeights</i> and <i>initBiases</i>.
 */
public enum Initializer {

	/**
	 * Useful if working on the CIFAR-10 classification task.
	 */
	XavierUniform {

		public void initWeights(float[][] weights) {

			final int prec_layer = weights[0].length;
			final int curr_layer = weights.length;
			final float factor = (float) (Math.sqrt(6.0 / (prec_layer + curr_layer)));

			for (int i = 0; i < weights.length; i++)
				for (int j = 0; j < weights[i].length; j++)
					weights[i][j] = random(-1.0, 1.0) * factor;
		}

		public void initBiases(float[] biases) {
			for (int i = 0; i < biases.length; i++)
				biases[i] = random(-1.0, 1.0);
		}
	},
	XavierNormal {

		public void initWeights(float[][] weights) {

			final int prec_layer = weights[0].length;
			final int curr_layer = weights.length;
			final double factor = Math.sqrt(2.0 / (prec_layer + curr_layer));
			final Random rnd = new Random();

			for (int i = 0; i < weights.length; i++)
				for (int j = 0; j < weights[i].length; j++)
					weights[i][j] = (float) (rnd.nextGaussian() * factor);
		}

		public void initBiases(float[] biases) {
			for (int i = 0; i < biases.length; i++)
				biases[i] = random(-1.0, 1.0);
		}
	},
	/**
	 * Useful if using ReLu.
	 */
	Kaiming {

		public void initWeights(float[][] weights) {

			final int prec_layer = weights[0].length;
			final float factor = (float) (Math.sqrt(prec_layer / 2));

			for (int i = 0; i < weights.length; i++)
				for (int j = 0; j < weights[i].length; j++)
					weights[i][j] = random(-1.0, 1.0) * factor;
		}

		public void initBiases(float[] biases) {
			for (int i = 0; i < biases.length; i++)
				biases[i] = random(-1.0, 1.0);
		}
	},
	/**
	 * Zero. No purpose.
	 */
	Zero {

		public void initWeights(float[][] weights) {
			for (int i = 0; i < weights.length; i++)
				for (int j = 0; j < weights[i].length; j++)
					weights[i][j] = 0.0f;
		}

		public void initBiases(float[] biases) {
			for (int i = 0; i < biases.length; i++)
				biases[i] = 0.0f;
		}
	};

	public abstract void initWeights(float[][] weights);

	public abstract void initBiases(float[] biases);

	private static float random(double min, double max) {
		return (float) ThreadLocalRandom.current().nextDouble(min, max);
	}
}
