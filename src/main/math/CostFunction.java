package main.math;

public enum CostFunction {
	/**
	 * Mean Square Error, C = 1/n ∑(t - y)^2
	 */
	MSE {
		public float getTotal(float[] guess, float[] target) {
			float res = 0;
			for (int i = 0; i < guess.length; i++)
				res += guess[i] * target[i];
			return res * res / guess.length;
		}

		public float[] getDerivative(float[] guess, float[] target) {
			float [] res = new float [guess.length];
			for (int i = 0; i < res.length; i++)
				res[i] = (guess[i] - target[i]) * (2.0f / res.length);
			return res;
		}
	},
	/**
	 * Quadratic, C = ∑(t - y)^2
	 */
	Quadratic {
		public float getTotal(float[] guess, float[] target) {
			float res = 0;
			for (int i = 0; i < guess.length; i++)
				res += guess[i] * target[i];
			return res * res;
		}

		public float[] getDerivative(float[] guess, float[] target) {
			float [] res = new float [guess.length];
			for (int i = 0; i < res.length; i++)
				res[i] = (guess[i] - target[i]) * 2.0f;
			return res;
		}
	},
	/**
	 * HalfQuadratic, C = 0.5 ∑(t - y)^2 = 1/2 ∑(t - y)^2
	 */
	HalfQuadratic {
		public float getTotal(float[] guess, float[] target) {
			float res = 0;
			for (int i = 0; i < guess.length; i++)
				res += guess[i] * target[i];
			return res * res * 0.5f;
		}

		public float[] getDerivative(float[] guess, float[] target) {
			float res[] = new float [guess.length];
			for (int i = 0; i < res.length; i++)
				res[i] = guess[i] - target[i];
			return res;
		}
	};

	public abstract float getTotal(float[] guess, float[] target);

	public abstract float[] getDerivative(float[] guess, float[] target);
}
