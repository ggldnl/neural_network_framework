package main.math;

/**
 * Each ActivationFunction provides the <i>execute</i> and <i>derivative</i>
 * methods, which return respectively the output of the function and of the its
 * derivative. To add a new ActivationFunction just create a new entry in the
 * enum and implement the two methods.
 */
public enum ActivationFunction {

	Sigmoid {
		public float execute(float z) {
			return 1 / (1 + (float) Math.exp(-z));
		}

		public float derivative(float z) {
			float s = execute(z);
			return s * (1 - s);
		}
	},
	TanH {
		public float execute(float z) {
			return (float) ((Math.exp(z) - Math.exp(-z)) / (Math.exp(z) + Math.exp(-z)));
		}

		public float derivative(float z) {
			float t = execute(z);
			return 1 - t * t;
		}
	},
	ReLU {
		public float execute(float z) {
			return z <= 0 ? 0 : z;
		}

		public float derivative(float z) {
			return z <= 0 ? 0 : 1;
		}
	},
	Leaky_ReLU {
		public float execute(float z) {
			return z <= 0 ? 0.01f * z : z;
		}

		public float derivative(float z) {
			return z <= 0 ? 0.01f : 1.0f;
		}
	},
	ArcTan {
		public float execute(float z) {
			return (float) Math.atan(z);
		}

		public float derivative(float z) {
			return 1 / (z * z + 1);
		}
	};

	public abstract float execute(float z);

	public abstract float derivative(float z);

	/**
	 * Applies the activation function over the array, saving the result in a new
	 * array and returning it.
	 */
	public float[] execute(float[] z) {
		float[] res = new float[z.length];
		for (int i = 0; i < res.length; i++)
			res[i] = execute(z[i]);
		return res;
	}

	/**
	 * Applies the derivative of the activation function over the array, saving the
	 * result in a new array and returning it.
	 */
	public float[] derivative(float[] z) {
		float[] res = new float[z.length];
		for (int i = 0; i < res.length; i++)
			res[i] = derivative(z[i]);
		return res;
	}

	/**
	 * Applies the activation function over the array (in-place).
	 */
	public void applyActivation(float[] z) {
		for (int i = 0; i < z.length; i++)
			z[i] = execute(z[i]);
	}

	/**
	 * Applies the derivative of the activation function over the array (in-place).
	 */
	public void applyDerivative(float[] z) {
		for (int i = 0; i < z.length; i++)
			z[i] = derivative(z[i]);
	}

	// static methods

	/**
	 * Applies the activation function passed as argument over the array, saving the
	 * result in a new array and returning it.
	 */
	public static float[] execute(float[] z, ActivationFunction activationFunction) {
		float[] res = new float[z.length];
		for (int i = 0; i < res.length; i++)
			res[i] = activationFunction.execute(z[i]);
		return res;
	}

	/**
	 * Applies the derivative of the activation function passed as argument over the
	 * array, saving the result in a new array and returning it.
	 */
	public static float[] derivative(float[] z, ActivationFunction activationFunction) {
		float[] res = new float[z.length];
		for (int i = 0; i < res.length; i++)
			res[i] = activationFunction.derivative(z[i]);
		return res;
	}

	/**
	 * Applies the activation function passed as argument over the array (in-place).
	 */
	public static void applyActivation(float[] z, ActivationFunction activationFunction) {
		for (int i = 0; i < z.length; i++)
			z[i] = activationFunction.execute(z[i]);
	}

	/**
	 * Applies the derivative of the activation function passed as argument over the
	 * array (in-place).
	 */
	public static void applyDerivative(float[] z, ActivationFunction activationFunction) {
		for (int i = 0; i < z.length; i++)
			z[i] = activationFunction.derivative(z[i]);
	}
};