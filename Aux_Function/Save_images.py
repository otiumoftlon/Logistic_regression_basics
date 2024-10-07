import matplotlib.pyplot as plt 
import seaborn as sns

def save_scatter_plot(data, x, y, hue='none', filename='scatter_plot.png', title='none', palette='deep'):
    """
    Generates and saves a scatter plot using seaborn and matplotlib.

    Parameters:
    df: Data general
    x (list or array): Data for the X-axis.
    y (list or array): Data for the Y-axis.
    hue: Color of hue
    filename (str): The filename to save the image. Default is 'scatter_plot.png'.
    title (str): Title of the scatter plot.
    xlabel (str): Label for the X-axis.
    ylabel (str): Label for the Y-axis.
    """
    # Set the style for seaborn
    sns.set_theme(style="whitegrid")
    # Create the scatter plot using seaborn
    plt.figure(figsize=(8, 6))  # Set the figure size
    sns.scatterplot(data=data,x=x, y=y,hue=hue, palette=palette)
    # Add plot title and axis labels
    plt.title(title)
    # Save the plot to a file
    plt.savefig(filename)

    # Show the plot (optional)
    plt.show()