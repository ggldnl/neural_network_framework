package main.test.mnist;

import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;

import main.training.DataSet;
import main.training.Match;

/**
 * This class incorporates the logic for a Dataset object creation starting from
 * the binary files containing the MNIST dataset. There isn't much to see here,
 * we've just applied what it is said to do on the MNIST dataset site.
 * 
 * http://yann.lecun.com/exdb/mnist/
 */
public class DigitDataSet extends DataSet {

	private static final int IMG_FILE_MAGIC_INT = 2051;
	private static final int LABEL_FILE_MAGIC_INT = 2049;

	@Override
	public DataSet createSet(File dataFile, File labelFile) {

		DataSet set = new DigitDataSet();

		try (DataInputStream image_stream = new DataInputStream(new FileInputStream(dataFile));
				DataInputStream label_stream = new DataInputStream(new FileInputStream(labelFile));) {

			if (image_stream.readInt() != IMG_FILE_MAGIC_INT)
				throw new IOException(String.format("Unknown file format for: %s.", dataFile.getName()));

			if (label_stream.readInt() != LABEL_FILE_MAGIC_INT)
				throw new IOException(String.format("Unknown file format for: %s.", labelFile.getName()));

			int nImages = image_stream.readInt();
			int nLabels = label_stream.readInt();

			if (nImages != nLabels)
				throw new IOException(
						String.format("File %s and file %s contains data for a different number of images.",
								dataFile.getName(), labelFile.getName()));

			int rows = image_stream.readInt();
			int cols = image_stream.readInt();

			for (int i = 0; i < nImages; i++) {

				byte[] data = new byte[rows * cols];
				float[] img = new float[rows * cols];

				image_stream.read(data, 0, data.length);
				for (int d = 0; d < img.length; d++)
					img[d] = (data[d] & 255) / 255.0f;

				int label = label_stream.readByte();

				set.addMatch(new Match(rows, cols, img, label));
			}

		} catch (Exception e) {
			e.printStackTrace();
		}

		return set;
	}
	
	@Override
	public DataSet createSet (File dataFile, File labelFile, int matches) {

		if (matches < 0)
			throw new IllegalArgumentException ("The number of matches in the dataset must be positive.");
		
		DataSet set = new DigitDataSet();

		try (DataInputStream image_stream = new DataInputStream(new FileInputStream(dataFile));
				DataInputStream label_stream = new DataInputStream(new FileInputStream(labelFile));) {
			
			if (image_stream.readInt() != IMG_FILE_MAGIC_INT)
				throw new IOException(String.format("Unknown file format for: %s.", dataFile.getName()));

			if (label_stream.readInt() != LABEL_FILE_MAGIC_INT)
				throw new IOException(String.format("Unknown file format for: %s.", labelFile.getName()));

			int nImages = image_stream.readInt();
			int nLabels = label_stream.readInt();

			if (nImages != nLabels)
				throw new IOException(
						String.format("File %s and file %s contains data for a different number of images.",
								dataFile.getName(), labelFile.getName()));

			if (nImages < matches)
				throw new IllegalArgumentException("The number of matches given exceeds the number of elements in the file.");
			
			int rows = image_stream.readInt();
			int cols = image_stream.readInt();

			for (int i = 0; i < matches; i++) {

				byte[] data = new byte[rows * cols];
				float[] img = new float[rows * cols];

				image_stream.read(data, 0, data.length);
				for (int d = 0; d < img.length; d++)
					img[d] = (data[d] & 255) / 255.0f;

				int label = label_stream.readByte();

				set.addMatch(new Match(rows, cols, img, label));
			}

		} catch (Exception e) {
			e.printStackTrace();
		}

		return set;
	}
	
	@Override
	public DataSet createSet(File dataFile) {

		DataSet set = new DigitDataSet();

		try (DataInputStream image_stream = new DataInputStream(new FileInputStream(dataFile));) {

			if (image_stream.readInt() != IMG_FILE_MAGIC_INT)
				throw new IOException(String.format("Unknown file format for: %s.", dataFile.getName()));

			int nImages = image_stream.readInt();

			int rows = image_stream.readInt();
			int cols = image_stream.readInt();

			for (int i = 0; i < nImages; i++) {

				byte[] data = new byte[rows * cols];
				float[] img = new float[rows * cols];

				image_stream.read(data, 0, data.length);
				for (int d = 0; d < img.length; d++)
					img[d] = (data[d] & 255) / 255.0f;

				set.addMatch(new Match(rows, cols, img));
			}

		} catch (Exception e) {
			e.printStackTrace();
		}

		return set;
	}
	
	@Override
	public DataSet createSet(File dataFile, int matches) {

		if (matches < 0)
			throw new IllegalArgumentException ("The number of matches in the dataset must be positive.");
		
		DataSet set = new DigitDataSet();

		try (DataInputStream image_stream = new DataInputStream(new FileInputStream(dataFile));) {

			if (image_stream.readInt() != IMG_FILE_MAGIC_INT)
				throw new IOException(String.format("Unknown file format for: %s.", dataFile.getName()));

			int nImages = image_stream.readInt();
			
			if (nImages < matches)
				throw new IllegalArgumentException("The number of matches given exceeds the number of elements in the file.");
				
			int rows = image_stream.readInt();
			int cols = image_stream.readInt();

			for (int i = 0; i < matches; i++) {

				byte[] data = new byte[rows * cols];
				float[] img = new float[rows * cols];

				image_stream.read(data, 0, data.length);
				for (int d = 0; d < img.length; d++)
					img[d] = (data[d] & 255) / 255.0f;

				set.addMatch(new Match(rows, cols, img));
			}

		} catch (Exception e) {
			e.printStackTrace();
		}

		return set;
	}
}
