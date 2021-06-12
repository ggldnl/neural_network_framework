package main;

/**
 * Listener for printing network informations on the console.
 */
public class ConsoleLogger {

	private static final int progressBarLength = 25;
	private static final char progressBarFill = '/'; /* '\u2588'; */
	private static final char progressBarEmpty = '.'; /* ' '; */

	private Network network;
	private boolean prev_statusbar = false;

	public ConsoleLogger(Network network) {
		this.network = network;
	}

	/**
	 * Creates the progressBar and returns it as a string. We must use \r to put the
	 * caret to the start of the line before printing out the string.
	 * 
	 * @param percentage boolean to determine whether to show the percentage in the
	 *                   string or not
	 * @param ratio      boolean to determine whether to show currentValue/maxValue
	 *                   in the string next to the percentage.
	 * @return the progress bar string.
	 */
	protected static String progressBar(int currentValue, int maxValue, boolean percentage, boolean ratio) {

		int currentProgressBarIndex = (int) Math.ceil(((double) progressBarLength / maxValue) * currentValue);
		StringBuilder sb = new StringBuilder();

		if (ratio)
			sb.append(String.format("%d/%d\t", currentValue, maxValue));

		if (percentage) {
			double percent = (100 * currentProgressBarIndex) / (double) progressBarLength;
			String formattedPercent = String.format("%6.2f %%\t", percent < 100.0d ? percent : 100.0d);
			sb.append(formattedPercent);
		}

		for (int progressBarIndex = 0; progressBarIndex < progressBarLength; progressBarIndex++)
			sb.append(currentProgressBarIndex <= progressBarIndex ? progressBarEmpty : progressBarFill);

		return sb.toString();
	}

	protected static String progressBar(int currentValue, int maxValue) {
		return progressBar(currentValue, maxValue, false, true);
	}

	/**
	 * Prints network info.
	 */
	public void update() {
		if (network.state != Network.State.training && prev_statusbar) {
			prev_statusbar = false;
			System.out.println();
		}
		switch (network.state) {
		case training:
			System.out.printf("\rTraining...\tEpoch %d/%d\tProgress %s", network.current_epoch, network.total_epoch,
					progressBar(network.current_match, network.total_matches));
			prev_statusbar = true;
			break;
		case saved:
			System.out.printf("Saved successfully in <%s>\n", network.file.getName());
			break;
		case not_saved:
			System.out.printf("Unable to save in <%s>\n", network.file.getName());
			break;
		case restored:
			System.out.printf("Restored successfully from <%s>\n", network.file.getName());
			break;
		case not_restored:
			System.out.printf("Unable to restore from <%s>\n", network.file.getName());
			break;
		default: // executing
		}
	}
}
