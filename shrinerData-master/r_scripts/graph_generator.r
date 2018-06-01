# Load the libraries to be used in the session
library(dplyr)
library(plotly)

# Loading the .csv file based on the patient ID number. 
# Get into the sliced data folder to read patient data, where patients are organized
cat("What is the Patient ID?: ");
id_num <- readLines("stdin", n = 1);

#Read in the file for the patient, each will be pre-processed in Python.
get_file <- function() {
  home_path <- (".././r_scripts")
  setwd(".././Sliced_Data")
  id_num <- paste(id_num, ".csv", sep = "")
  file <- paste(getwd(), id_num, sep = "/")
  setwd(home_path)
  file
}

#Create a folder for every patient based on ID number, but do not overwrite the files
create_folder <- function(id_num) {
  setwd(".././visualizations/By_Patient")
  print(getwd())
  print(dir.create(file.path(getwd(), id_num), showWarnings = FALSE))
  folder_path <- paste(getwd(), id_num, sep = "/")
  folder_path
}

#Create a folder to store the "full" data visuals
create_folder_all <- function(folder_parent) {
  setwd(folder_parent)
  print(getwd())
  print(dir.create(file.path(getwd(), "Data_All"), showWarnings = FALSE))
  folder_path <- paste(getwd(), "Data_All", sep = "/")
  folder_path
}

#Create folder to store afflicted-side-only graphs
create_folder_afflicted <- function(folder_parent) {
  setwd(folder_parent)
  print(getwd())
  print(dir.create(file.path(getwd(), "Data_Afflicted"), showWarnings = FALSE))
  folder_path <- paste(getwd(), "Data_Afflicted", sep = "/")
  folder_path
}

#Wrap functions to write the data into a frame, based on the inputs given.

#GDI vs KneeFlexionPeak
create_graph_knee <- function(data, folder, title, filename) {
  setwd(folder)

  ax <- list(
       title = "Knee Flexion Peak",
       zeroline = FALSE,
       showline = TRUE,
       showticklabels = TRUE,
       showgrid = TRUE
   )

  plot <- plot_ly(
  data, x = ~KneeFlexionPeak, y =  ~GDI, type = 'scatter',
  text = ~paste('Study:', StudyType, '<br>Test Date:', TestDate, '<br>Body Side:', MotionParams.Side, '<br>Age:', AgeTest),
  color = ~AgeTest) %>%
  add_lines(y = ~fitted(loess(GDI ~ KneeFlexionPeak)),
            line = list(color = '#07A4B5'),
            name = "Loess Trend", showlegend = TRUE) %>%
  layout(title = title, xaxis = ax)

  # Store the generated file(s) into a folder within the R_Visualizations Folder
  htmlwidgets::saveWidget(plot, filename)
}

#GDI vs WalkingSpeed
create_walking_graph <- function(data, folder, title, filename) {
  ax <- list(
       title = "Walking Speed (m/s)",
       zeroline = FALSE,
       showline = TRUE,
       showticklabels = TRUE,
       showgrid = TRUE
   )

  plot <- plot_ly(
  data, x = ~WalkingSpeedMetersPerSec, y =  ~GDI, type = 'scatter',
  text = ~paste('Study:', StudyType, '<br>Test Date:', TestDate, '<br>Body Side:', MotionParams.Side, '<br>Age:', AgeTest),
  color = ~AgeTest) %>%
  add_lines(y = ~fitted(loess(GDI ~ WalkingSpeedMetersPerSec)),
            line = list(color = '#07A4B5'),
            name = "Loess Trend", showlegend = TRUE) %>%
  layout(title = title, xaxis = ax)

  # Store the generated file(s) into a folder within the R_Visualizations Folder
  htmlwidgets::saveWidget(plot, filename)
}

#Walking Speed vs Step Length 
create_walking_step_graph <- function(data, folder, title, filename) {
  ax <- list(
       title = "Step Length (meters)",
       zeroline = FALSE,
       showline = TRUE,
       showticklabels = TRUE,
       showgrid = TRUE
   )

  ay <- list(
       title = "Walking Speed (m/s)",
       zeroline = FALSE,
       showline = TRUE,
       showticklabels = TRUE,
       showgrid = TRUE
   )
  plot <- plot_ly(
  data, x = ~StepLengthM, y =  ~WalkingSpeedMetersPerSec, type = 'scatter',
  text = ~paste('Study:', StudyType, '<br>Test Date:', TestDate, '<br>Body Side:', MotionParams.Side, '<br>Age:', AgeTest),
  color = ~AgeTest) %>%
  add_lines(y = ~fitted(loess(WalkingSpeedMetersPerSec ~ StepLengthM)),
            line = list(color = '#07A4B5'),
            name = "Loess Trend", showlegend = TRUE) %>%
  layout(title = title, xaxis = ax, yaxis= ay)

  # Store the generated file(s) into a folder within the R_Visualizations Folder
  htmlwidgets::saveWidget(plot, filename)

}

#Make a subset of the data, so that we can probe specifically afflicited sides

#Read the file as a .csv after getting it. 
file <- get_file()
data <- read.csv(file)

#Check for consistency
print(data)

#Create a folder in the by_patient folder, 
#initialized to put data into. 
folder_patient <- create_folder(id_num)
folder_patient_all <- create_folder_all(folder_patient)
folder_patient_afflicted <- create_folder_afflicted(folder_patient)

#Create Data for ALL Sides that have been effected 
# =========================================================================
#Generate Graph Title
graph_title <- paste("PID", id_num, "Trends")
#Generate filename Title
filename_knee <- paste(id_num, "_Knee_v_GDI.html")
filename_walking <- paste(id_num, "_Walking_v_GDI.html")
filename_walking_steps <- paste(id_num, "_Walking_vs_Steps.html")

#Create Graphs For all sides of the body
create_graph_knee(data, folder_patient_all, graph_title, filename_knee)
create_walking_graph(data, folder_patient_all, graph_title, filename_walking)
create_walking_step_graph(data, folder_patient_all, graph_title,
filename_walking_steps)
# ==========================================================================

#Create Dataframe based on the spliced subset of data

#Get Side Affected
side_affected <- data$Procedures.Side[[1]]

#Write Side affected dataframe
data_afflicited <- split(data,data$MotionParams.Side)[[side_affected]]

#Set new working dir, then move through and create graphs :) 
# =========================================================================
setwd(folder_patient_afflicted)
print(getwd())

#New Graph Titles
graph_title_afflicted <- paste("PID", id_num, "Trends (Afflicted Side)")

#New filenames 
filename_knee_afflicted <- paste(id_num, "_Knee_v_GDI_Afflicted.html")
filename_walking_afflicted <- paste(id_num, "_Walking_v_GDI_Afflicted.html")
filename_steps_afflicted <- paste(id_num, "_Walking_vs_Steps_Afflicted.html")


#Create Graphs
create_graph_knee(data_afflicited, folder_patient_afflicted,
graph_title_afflicted, filename_knee_afflicted)

create_walking_graph(data_afflicited, folder_patient_afflicted,
graph_title_afflicted, filename_walking_afflicted)

create_walking_step_graph(data_afflicited, folder_patient_afflicted,
graph_title_afflicted, filename_steps_afflicted)