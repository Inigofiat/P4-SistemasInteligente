# -----------------------------------------------------------------------------
# This code snippet was generated for educational purposes as part of the
# Intelligent Systems course at the University of Deusto. The code has been
# created with assistance from ChatGPT version 3.5 and GitHub Copilot.
#
# The code is released under the Creative Commons License and is provided
# for free use and modification by the programming and development community.
#
# This script was generated in April 2024, the year when the Athletic Club de
# Bilbao won the 25th King's Cup.
# -----------------------------------------------------------------------------

# Creates a scatter plot of each attribute except the target and stores each
# plot as an image.
plot.data.distribution <- function(data, target, folder = "../data/images/") {
  # Get the names of attributes except the "target"
  attributes <- setdiff(names(data), target)

  # Create a plot for each attribute and save it as a PNG image
  for (i in 1:length(attributes)) {
    attribute <- attributes[i]
    plot <- ggplot(data, aes_string(x = attribute, y = target)) +
      geom_point(color = "blue") +
      theme_bw() +
      labs(x = attribute, y = target)
    filename <- paste0(folder, "scatter-plot-", target, "-vs-", attribute, ".png")
    ggsave(filename, plot, width = 6, height = 4, dpi = 300)
  }
}

# Creates a scatter plot of each attribute except the target and stores each
# plot as an image.
plot.data.frequency <- function(data, folder = "../data/images/") {
  # Get the names of attributes
  attributes <- names(data)

  # Create a plot for each attribute and save it as a PNG image
  for (i in 1:length(attributes)) {
    attribute <- attributes[i]

    plot <- ggplot(data, aes_string(x = attribute, fill = attribute)) +
      geom_bar() +
      geom_text(stat = "count", aes(label = ..count..), vjust = -0.25, position = position_stack(vjust = 0.5)) +
      labs(x = attribute, y = "Frequency")
    filename <- paste0(folder, "frequency-", attribute, ".png")
    ggsave(filename, plot, width = 8, height = 4, dpi = 300)
  }
}
