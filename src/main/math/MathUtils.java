package main.math;

/**
 * Math operations over vectors and matrices.
 */
public class MathUtils {

	private static void checkSameSize(float[][] m1, float[][] m2) {
		if (m1.length != m2.length)
			throw new IllegalArgumentException(String.format("m1.rows[%s] != m2.rows[%s].", m1.length, m2.length));
		if (m1[0].length != m2[0].length)
			throw new IllegalArgumentException(
					String.format("m1.cols[%s] != m2.cols[%s].", m1[0].length, m2[0].length));
	}

	private static void checkSameSize(float[] v1, float[] v2) {
		if (v1.length != v2.length)
			throw new IllegalArgumentException(String.format("v1.length[%s] != v2.length[%s].", v1.length, v2.length));
	}

	public static float dot(float[] a, float[] b) {
		if (a.length != b.length)
			throw new IllegalArgumentException("a.length[" + a.length + "] != b.length[" + b.length + "].");

		float res = 0;
		for (int i = 0; i < a.length; i++)
			res += (a[i] * b[i]);
		return res;
	}

	public static float[][] mul(float[][] m1, float[][] m2) {
		if (m1[0].length != m2.length)
			throw new IllegalArgumentException(
					"first_matrix.cols[" + m1[0].length + "] != second_matrix.rows[" + m2.length + "].");

		float[][] product = new float[m1.length][m2[0].length];
		for (int i = 0; i < m1.length; i++) {
			for (int j = 0; j < m2[0].length; j++) {
				for (int k = 0; k < m1[0].length; k++) {
					product[i][j] += m1[i][k] * m2[k][j];
				}
			}
		}
		return product;
	}

	public static float[] mul(float[] vector, float[][] matrix) {
		if (vector.length != matrix.length)
			throw new IllegalArgumentException(
					"first_matrix.cols[" + vector.length + "] != second_matrix.rows[" + matrix.length + "].");

		float[] product = new float[matrix[0].length];
		for (int i = 0; i < product.length; i++)
			for (int j = 0; j < matrix.length; j++)
				product[i] += vector[j] * matrix[j][i];
		return product;
	}

	public static float[] mul(float[][] matrix, float[] vector) {
		if (matrix[0].length != vector.length)
			throw new IllegalArgumentException(
					"first_matrix.cols[" + matrix[0].length + "] != second_matrix.rows[" + vector.length + "].");

		float[] product = new float[matrix.length];
		for (int i = 0; i < product.length; i++)
			product[i] = dot(vector, matrix[i]);
		return product;
	}

	/*
	 * The first vector must be a column vector, and the second a row vector,
	 * because otherwise we have a dot product and there is already a method for
	 * that.
	 */
	public static float[][] mul(float[] v1, float[] v2) {
		float[][] product = new float[v1.length][v2.length];
		for (int i = 0; i < v1.length; i++) {
			for (int j = 0; j < v2.length; j++) {
				product[i][j] = v1[i] * v2[j];
			}
		}
		return product;
	}

	public static void map(float[][] matrix, float val) {
		for (int i = 0; i < matrix.length; i++)
			for (int j = 0; j < matrix[i].length; j++)
				matrix[i][j] *= val;
	}

	public static float[] applyDerivative(ActivationFunction sigma, float[] arr) {
		float[] res = new float[arr.length];
		for (int i = 0; i < res.length; i++)
			res[i] = sigma.derivative(arr[i]);
		return res;
	}

	public static float[][] applyDerivative(ActivationFunction sigma, float[][] arr) {
		float[][] res = new float[arr.length][arr[0].length];
		for (int i = 0; i < res.length; i++)
			for (int j = 0; j < res[i].length; j++)
				res[i][j] = sigma.derivative(arr[i][j]);
		return res;
	}

	public static float[][] transpose(float[][] m) {
		float[][] res = new float[m[0].length][m.length];
		for (int i = 0; i < res.length; i++)
			for (int j = 0; j < res[i].length; j++)
				res[i][j] = m[j][i];
		return res;
	}

	public static float[][] sub(float[][] m1, float[][] m2) {
		checkSameSize(m1, m2);

		float[][] res = new float[m1.length][m1[0].length];
		for (int i = 0; i < res.length; i++)
			for (int j = 0; j < res[i].length; j++)
				res[i][j] = m1[i][j] - m2[i][j];
		return res;
	}

	public static float[] sub(float[] v1, float[] v2) {
		checkSameSize(v1, v2);

		float[] res = new float[v1.length];
		for (int i = 0; i < res.length; i++)
			res[i] = v1[i] - v2[i];
		return res;
	}

	public static float[][] sum(float[][] m1, float[][] m2) {
		checkSameSize(m1, m2);

		float[][] res = new float[m1.length][m1[0].length];
		for (int i = 0; i < res.length; i++)
			for (int j = 0; j < res[i].length; j++)
				res[i][j] = m1[i][j] + m2[i][j];
		return res;
	}

	public static float[] sum(float[] v1, float[] v2) {
		checkSameSize(v1, v2);

		float[] res = new float[v1.length];
		for (int i = 0; i < res.length; i++)
			res[i] = v1[i] + v2[i];
		return res;
	}

	/**
	 * Computes the element-wise product between dCdO and the derivative of the
	 * output vector. Used in the backpropagation process.
	 */
	public static float[] dCdI(ActivationFunction sigma, float[] output, float[] dCdO) {
		checkSameSize(output, dCdO);
		float[] res = new float[output.length];
		for (int i = 0; i < res.length; i++)
			res[i] = dCdO[i] * sigma.derivative(output[i]);
		return res;
	}

	/**
	 * Utility method to print a matrix (debug purposes).
	 */
	public static void printMatrix(float[][] matrix) {
		StringBuilder builder = new StringBuilder();
		for (int i = 0; i < matrix.length; i++)
			for (int j = 0; j < matrix[i].length; j++) {
				builder.append(matrix[i][j]);
				if (j < matrix[i].length - 1)
					builder.append('\t');
				else
					builder.append('\n');
			}
		builder.append('\n');
		System.out.println(builder.toString());
	}
}
