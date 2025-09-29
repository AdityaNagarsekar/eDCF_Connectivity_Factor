# FRACTAL PROJECT

## Table of Contents

- [Overview](#overview)
- [Version Compatibility](#version-compatibility)
- [Project Architecture](#project-architecture)
  - [Core Components](#core-components)
  - [Data Flow](#data-flow)
  - [Web Interface](#web-interface)
- [Template Guide](#template-guide)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
  - [Command Line Interface](#command-line-interface)
  - [Web Interface](#web-interface-1)
- [Analysis Features](#analysis-features)
- [Considerations](#considerations)
- [Resources](#resources)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

The Fractal Project is a comprehensive suite for analyzing geometric structures, fractals, and point clouds. It provides:

1. **Data Generation**: Create and manipulate multi-dimensional geometric structures
2. **Grid Analysis**: Generate classification-based grid of point cloud data
   - Uses ML algorithms to classify points into grid cells
3. **Boundary Detection**: Extract boundaries from classification results
   - Identifies transitions between classified regions
4. **Fractal Analysis**: Estimate fractal properties
   - Box-counting dimension
   - Hausdorff dimension
5. **Connectivity Analysis**: Analyze topological relationships
   - Connectivity assessment
   - Structural analysis
6. **Visualization**: Generate plots and reports for all analyses

## Version Compatibility

### Core Requirements (Version 2.6.1)
- Python 3.8+
- Django ≥5.1
- numpy ≥1.20
- matplotlib ≥3.5
- scipy ≥1.7
- scikit-learn ≥1.0
- pandas ≥1.3

## Project Architecture

### Entry Points

1. **manage.py** (Frontend)
   - Django management interface
   - Provides web interface functionality
   - Used by: Web Interface
   - Entry point for web operations

2. **Driver (`Driver.py`)**
   - Main application entry point
   - Orchestrates the analysis pipeline
   - Used by: manage.py
   - Entry point for command-line operations

### Data Generation

1. **DynamicDataGenerator**
   - Creates and manipulates geometric structures
   - Generates point cloud data
   - Used by: Driver
   - Entry point for data creation

### Core Algorithms

1. **Manager (`Manager.py`)**
   - Central orchestrator for data processing
   - Handles grid computation, boundary extraction, and visualization
   - Manages file I/O and result persistence
   - Orchestrates analysis pipeline
   - Uses: GridGenerator, BoundaryExtractor, FractalDetector, Connectivity

2. **GridGenerator**
   - Creates grid representations of point clouds
   - Uses classification algorithms for grid generation
   - Used by: Manager
   - First step in analysis pipeline

3. **BoundaryExtractor**
   - Extracts boundaries from grid classifications
   - Identifies transitions between regions
   - Used by: Manager
   - Second step in analysis pipeline

4. **FractalDetector**
   - Computes fractal dimensions
   - Implements box-counting and Hausdorff methods
   - Used by: Manager
   - Third step in analysis pipeline

5. **Connectivity**
   - Analyzes point cloud connectivity
   - Evaluates structural relationships
   - Used by: Manager
   - Fourth step in analysis pipeline

### Standalone Components

1. **ForceGrid**
   - Handles grid object forcing
   - Manages grid interactions
   - Used by: Driver
   - Standalone grid manipulation tool
   - Directly manipulates grid objects

2. **TopologicalDimension**
   - Calculates topological properties
   - Uses NCubeNeighbour and Weight calculations
   - Used by: Driver
   - Standalone topological analysis tool
   - Performs independent topological analysis
   - Uses: NCubeNeighbour, Weight

### Supporting Modules

1. **NCubeNeighbour**
   - Implements neighbourhood calculations
   - Used by: BoundaryExtractor, Connectivity, TopologicalDimension
   - Provides neighbourhood analysis for boundary detection and connectivity

2. **Weight**
   - Handles weighting calculations
   - Used by: Connectivity
   - Provides weight calculations for connectivity analysis

3. **Interpreter (`Interpreter.py`)**
   - Generates comprehensive analysis reports
   - Compiles results from various analyses
   - Creates standardized output formats
   - Used by: Driver
   - Final step in output generation

4. **Save**
   - Manages result persistence
   - Handles file I/O operations
   - Used by: Interpreter
   - Final step in result persistence

### Data Flow

1. **Input Processing**
   - **DynamicDataGenerator**
     - Output: `Datapoints.npy` - Raw point cloud data
   
   - **Driver**
     - Output: `Grid_Info.npy` - Grid parameters
     - Output: `Spacing_Allocation_Details.npy` - Grid spacing details

2. **Analysis Pipeline**

   - **Grid Generation**
     - Input: `Datapoints.npy`
     - Output: 
       - `Grid_Points.npy` - Grid representation of point cloud
       - `Divided_Grid_Bounds.npy` - Divided grid boundaries
       - `Grid_Bounds.npy` - Overall grid boundaries

   - **Boundary Detection**
     - Input: `Grid_Points.npy`
     - Output: `Neighbour_Boundary.npy` - Boundary points between structures
     - Output: `Boundary_Type.npy` - Type of boundary detected

   - **Fractal Analysis**
     - Input: `Neighbour_Boundary.npy`
     - Output: `Dimensions.npy` - Computed fractal dimensions

   - **Connectivity Analysis**
     - Input: `Neighbour_Boundary.npy`
     - Output: 
       - `Connectivity_Factors.npy` - Connectivity measurements
       - `Weights_Boundaries.npy` - Weighting factors for boundaries

3. **Output Generation**
   - **Plots** (Generated by Manager)
     - Various `.png` files for visualization

   - **Data Files**
     - All `.npy` files containing analysis results

   - **Reports**
     - `Report.txt`: Generated by Interpreter
     - Contains analysis results and interpretations

4. **Standalone Operations**

   - **ForceGrid**
     - Input: `Datapoints.npy`
     - Performs: Object gridding analysis
     - Output: 
       - `Force_Grid_Spacings.npy` - Modified grid spacings
       - Updated `Grid_Info.npy` - Updated grid parameters

   - **TopologicalDimension**
     - Input: `Grid_Info.npy`
     - Output: 
       - `Topological_Dimensions_Boundaries.npy` - Topological dimensions for boundaries
       - `Weighted_Topological_Dimensions_Boundaries.npy` - Weighted dimensions for boundaries
       - `Weights_Boundaries_Normalized.npy` - Normalized weights for boundaries
       - `Topological_Dimensions_Object.npy` - Topological dimensions for objects
       - `Weighted_Topological_Dimensions_Force.npy` - Weighted dimensions for forced grid
       - `Weights_Force_Normalized.npy` - Normalized weights for forced grid

5. **Performance Analysis**
   - **TimeAnalysis**
     - Output:
       - `Time_Grid.npy` - Grid generation timing
       - `Time_Boundary.npy` - Boundary detection timing
       - `Time_Fractal.npy` - Fractal analysis timing
       - `Time_Connectivity.npy` - Connectivity analysis timing
       - `Time_Force_Grid.npy` - Force grid timing
       - `X_Grid_Space.npy` - Grid space scaling metrics
       - `X_Grid_Divide.npy` - Grid divide scaling metrics

### Web Interface

The Driver Interface is a comprehensive control panel for managing fractal analysis operations. It features a Control Center with multiple sections for different types of operations.

#### Control Center Layout

1. **Data Generation and Preparation**
   - **Generate Data**
     - Creates new point cloud data
     - CAUTION: Will override previous data
   - **Linear Transform Data**
     - Performs global normalization
   - **Train Algorithm**
     - For classification purposes
   - **Calculate Grid Parameters**
     - Required for grid operations

2. **Grid Operations**
   - **Compute Classification Grid**
     - Requires algorithm training
   - **Extract Boundary**
     - Requires classified grid
   - **Force Grid Data**
     - Converts datapoints to grid points
   - **Direct Conversion**
     - CAUTION: Does not provide hatch feature
   - **Dynamic Spacing Allocation**
     - Adjusts spacing dynamically for data

3. **Analysis Operations**
   - **Assess Fractal Dimension (Boundaries)**
     - Requires extracted boundaries
   - **Calculate Connectivity Factor**
     - Requires extracted boundaries
   - **Estimate Topological Dimension**
     - Requires connectivity factor computations
   - **Estimate Fractal Dimension**
     - Requires fractal computations
   - **Analyse Range**
     - Analysis over percentage data range
   - **Analyse Connectivity Deterioration**
     - CAUTION: May take a long time

4. **Display Operations**
   - **Data Display**
     - Limited to 2D data
   - **Grid Display**
     - Requires classified grid
   - **Boundary Display**
     - Requires extracted boundaries
   - **Display Force Grid Data**
     - Requires force grid data
   - **Display Hatch Data**
     - Requires hatch data
   - **Display Range Analysis**
     - Requires range analysis
   - **Display Connectivity Deterioration**
     - Requires connectivity deterioration analysis
   - **Display Time Analysis**
     - Requires time analysis

5. **Save Operations**
   - **Save Force Data**
     - Saves force grid related data
   - **Save Deterioration Data**
     - Saves connectivity deterioration data
   - **Save**
     - Saves all global generated data
   - **Save Time Data**
     - Saves time analysis related data

6. **Data Fields**
   - **Algorithm**
     - Enter Python code for algorithm object
   - **Grid Spacing Scaling**
     - For time analysis
   - **Division Factor**
     - For dynamic spacing
   - **Bias**
     - Error bias for analysis
   - **Range of Divisions**
     - For time analysis (multiply)
   - **Force Identity**
     - Identity for force grid operations
   - **Grid Division Scale**
     - Scale for grid division
   - **Grid Spacing Scale**
     - Scale for grid spacing
   - **Range of Division Add**
     - Range for division addition

#### Button Status Indicators
- Green (active): Operation will be performed
- Blue (inactive): Operation is disabled
- Click: Toggles operation status

#### Important Notes

1. **Operation Requirements**
   - Most operations have prerequisites (indicated in button messages)
   - Critical operations are marked with CAUTION warnings
   - Long-running operations may take significant time to complete

2. **Data Management**
   - Use "Save" operations to preserve your work
   - The interface maintains state between sessions
   - Pressing `Enter` in a data field saves values for future sessions

3. **System Operations**
   - **Refresh Page Contents**: Resets the interface to its initial state
   - **Clear System**: Removes all unsaved data files
   - **Download Folder**: 
     - Deletes the folder from the server
     - Makes it available for download
     - Use with caution as this operation is irreversible

4. **Current Limitations**
   - Time Analysis functionality is not fully implemented
   - Some operations may have performance limitations in high dimensions
   - Data handling requires strict adherence to format requirements

## Template Guide

The Fractal Project provides template files to help you quickly implement data structures and algorithms without having to remember all the required parameters and syntax.

### Data Structures Template

The `Data_Structures_Template.txt` file contains ready-to-use templates for creating various geometric structures and datasets for fractal analysis.

#### How to Use Data Structures Templates

1. **Basic Usage**
   ```python
   from DynamicDataGenerator import DynamicDataGenerator
   from data_structures.Circle import Circle
   from data_structures.Spiral import Spiral
   
   # Copy template from Data_Structures_Template.txt
   circle = Circle(identity=1, radius=2.0, center=(0.0, 0.0), noise_rate=0.1)
   spiral = Spiral(identity=2, angle_start=0, angle_end=720, center=(0.0, 0.0), noise_rate=0.1)
   
   # Create generator with structures
   generator = DynamicDataGenerator(data_objects=[circle, spiral], num_points=[1000, 2000])
   generator.generate_data(linear_transform=True)
   ```

2. **Available Templates**
   - **Basic Geometric Shapes**:
     - Circle (2D): `Circle(identity=0, radius=0.0, center=(0.0, 0.0), noise_rate=0.0)`
     - Filled Circle: `Circle(identity=0, radius=0.0, center=(0.0, 0.0), noise_rate=0.0, filled_in=True)`
     - Spiral: `Spiral(identity=0, angle_start=0, angle_end=0, center=(0.0, 0.0), noise_rate=0.0)`
     - Sphere (3D): `Sphere(identity=0, radius=0.0, center=(0.0, 0.0, 0.0), noise_rate=0.0)`
     - Sphere (4D): `Sphere4D(identity=1, radius=0.0, center=(0.0, 0.0, 0.0, 0.0), noise_rate=1.0)`
   
   - **Fractal Structures**:
     - Barnsley Fern: `BarnsleyFern(identity=1, iterations=5000000, mask_points=5000000, min_distance=0.01)`
     - Mandelbrot Set: `MandelbrotSet(identity=1, iterations=5000000, mask_points=5000000, min_distance=0.01, power=11)`
     - Julia Set: `JuliaSet(identity=1, iterations=5000000, mask_points=5000000, min_distance=0.01)`
     - Sierpinski Triangle: `SierpinskiTriangle(identity=1, iterations=5000000, mask_points=5000000, min_distance=0.01)`
     - Sierpinski Carpet: `SierpinskiCarpet(identity=1, iterations=5000000, mask_points=5000000, min_distance=0.01)`
   
   - **Mathematical Curves**:
     - Sinusoidal Curve: `SinusoidalCurve(identity=1, amplitude=1.0, phase_difference=0.0, y_center=0.0, noise_rate=0.5)`
     - V-Shape: `VShapeGenerator(identity=1, angle=-45, shift=10.0, noise_rate=0.6, repetitions=5)`
   
   - **Real-world Datasets**:
     - Iris Setosa: `IrisSetosaData(identity=1, pca_components=2)`
     - Iris Versicolor: `IrisVersicolorData(identity=2, pca_components=2)`
     - Iris Virginica: `IrisVirginicaData(identity=3, pca_components=2)`
     - Android Data: `AndroidData(identity=1)`
   
   - **Utility Structures**:
     - Mask: `Mask(identity=0)`  # Must be called after a fractal data set

3. **Important Parameters**
   - `identity`: Unique positive integer identifying the structure
   - `noise_rate`: Amount of random noise to add (0.0 = no noise)
   - `iterations`: For fractal structures, controls detail level
   - `mask_points`: For fractal structures with Mask, number of points to exclude
   - `min_distance`: Minimum distance between points

### Algorithms Template

The `Algorithms_Template.txt` file provides templates for machine learning algorithms used in grid classification.

#### How to Use Algorithm Templates

1. **Basic Usage**
   ```python
   from algorithms.SVM import SVM
   
   # Copy template from Algorithms_Template.txt
   algorithm = SVM(C=[0.1, 1, 10], kernel=['rbf'], gamma=['scale','auto'], decision_function_shape=['ovo'])
   
   # Use in Driver
   driver = Driver()
   driver.algorithm = algorithm
   driver.algorithm_train_ctrl = True
   ```

2. **Available Templates**
   - **K-Nearest Neighbors**:
     ```python
     KNN(k_start=1, p_start=1, p_lim=5, k_lim=75, cvn=5, leaf_start=1, leaf_lim=50)
     ```
   
   - **Multi-layer Perceptron**:
     ```python
     MLP(hidden_layer_sizes=[(10, 10, 10)], activation=['tanh', 'relu', 'logistic'], 
         solver=['sgd', 'adam'], alpha=[0.0001, 0.001], 
         learning_rate=['constant', 'adaptive'], max_iter=[5000])
     ```
   
   - **Decision Tree**:
     ```python
     DecisionTree(criterion=['gini', 'entropy'], splitter=['best', 'random'], 
                 max_depth=[None, 5, 10, 20], min_samples_split=[2, 5, 10], 
                 min_samples_leaf=[1, 2, 4], max_features=[None, 'sqrt', 'log2'])
     ```
   
   - **Support Vector Machine**:
     ```python
     SVM(C=[0.1, 1, 10], kernel=['rbf'], gamma=['scale','auto'], 
         decision_function_shape=['ovo'])
     ```

3. **Parameter Optimization**
   - Parameters are specified as lists for grid search optimization
   - The system will automatically find the best combination
   - For faster results, narrow down parameter ranges

### Best Practices

1. **Data Structure Creation**
   - Always use unique positive integers for identity values
   - Set appropriate noise levels based on analysis needs
   - For fractal structures, balance iterations with performance

2. **Algorithm Selection**
   - SVM works well for most boundary detection tasks
   - KNN is faster but may be less accurate for complex boundaries
   - MLP provides good results for non-linear boundaries but takes longer to train
   - Decision Trees work well for categorical data

3. **Memory Considerations**
   - Be cautious with high iteration counts for fractal structures
   - Reduce point counts for high-dimensional data
   - Monitor memory usage when working with complex structures

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/i-to-the-power-i/fractals.git
   cd fractals
   ```

2. **Create Virtual Environment(Optional)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   .\venv\Scripts\activate  # Windows
   ```

3. **Install Dependencies**
   - Install dependencies by referring to version compatibility section

## Usage Guide

### Command Line Interface

1. **Set Driver __init__ values**
   ```python
   Driver.__init__() # Set all values in __init__
   ```

2. **Run Analysis**
   ```bash
   python3 Driver.py
   ```

### Web Interface

1. **Setup**
   - Create .env file with `SECRET_KEY` and `DEBUG` mode
   - Run migrations
     ```bash
     python3 manage.py migrate
     ```

2. **Start Server**
   ```bash
   python3 manage.py runserver
   ```

3. **Access Interface**
   - Open browser at `http://127.0.0.1:8000`
   - Upload data or use example datasets
   - Configure analysis parameters
   - View results and download reports

## Considerations

1. **Memory Usage**
   - Be cautious with high iteration counts for fractal structures
   - Reduce point counts for high-dimensional data
   - Monitor memory usage when working with complex structures

2. **Multi-core Processing**
   - Utilize multi-core processing for faster analysis
   - Adjust `core_ctrl` parameter in Driver
   - NOTE: Right now the method for multi-core processing is not working as expected in Connectivity and ForceGridObject for high point count and thus it is commented out but can be activated via lines @379 - 384 in Connectivity.py and @496 - 506 in ForceGridObject.py. Also comment out appropriate lines in the same files to disable single core processing.

3. **Error Points**
   - In Analysis, there can be floating point errors due to the precision of the calculations. We do not consider these points in our polynomial approximations.
   - To prevent these errors from occuring in the first place there is a rounding limit of 14 decimal places in ForceGridObject.py.
   - Bias is a tolerance term which goes to 0 tolerance as it approaches 1 and must be between 0-1.

## Resources

- [All Data](https://drive.google.com/drive/folders/1PDIkDLce8OtwUzXSLZ8Qc19X-9CO-_sz?usp=drive_link)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Authors: Dhruv Gupta, Vraj Shah, Aditya Nagarsekar, Harikrishnan N. B.
- Version: 2.6.2
