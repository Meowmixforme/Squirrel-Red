# Red Squirrel Population Prediction in Scotland

## Project Overview



<img width="500" height="500" alt="470070590-ccf8ebc6-914f-483c-bdd4-5ea52e24a3a1" src="https://github.com/user-attachments/assets/9f87bbbe-58d1-4ad4-b1e2-f46f552ab111" />




This project uses machine learning and spatial regression techniques to predict red squirrel populations across Scottish local authorities using mixed-source citizen science data. The system includes a two-stage duplicate detection algorithm and a Random Forest Regressor model to process and analyse data from the Scottish Squirrel Database.

## Key Features

- **Two-Stage Duplicate Detection:** Identifies and removes duplicate observations using both exact spatial matching and proximity-based detection that accounts for coordinate uncertainty
- **Spatial-Temporal Modelling:** Predicts future squirrel populations using Random Forest Regressor with spatial and temporal features
- **Interactive Visualisation:** Streamlit web application with dynamic maps, charts, and prediction tools
- **Uncertainty Quantification:** Provides confidence intervals for predictions to support conservation decision-making

  
<img width="929" height="901" alt="466092226-5d2d5f58-db8b-434a-a3ca-6e2df35a7db6" src="https://github.com/user-attachments/assets/90e564ab-adc5-4440-9528-a7566fde73d1" />



## Technologies Used

- **Python:** Primary programming language
- **MongoDB:** Database storage for squirrel observation records
- **Streamlit:** Web application framework for interactive visualisation
- **Scikit-learn:** Implementation of Random Forest Regressor model
- **GeoPandas:** Spatial data processing and analysis
- **Folium:** Interactive mapping
- **Plotly:** Statistical visualisations

## Dataset

The system uses the Scottish Squirrel Database obtained via the NBN Atlas, containing over 93,000 squirrel sightings recorded between 1905 and 2021. The data is filtered, cleaned, and processed to focus on red squirrels in Scotland from 2010-2021.

## Results

- Successfully identified and removed 5.36% of records as potential duplicates
- Achieved an R² score of 0.849 during cross-validation and 0.79 on the test set
- Identified current year density (50.8%) and previous year count (35.5%) as the most influential predictors
- Created an intuitive application for exploring population trends and generating future predictions


<img width="500" height="270" alt="466092660-ad141ba5-f1a9-4651-9ec2-0c79d5b95c7f" src="https://github.com/user-attachments/assets/19c3a003-8df8-4519-84fa-038a8316983d" />


## Installation and Usage

1. Clone this repository
2. Install required packages: `pip install -r requirements.txt`
3. Configure MongoDB connection in config.py
4. Run the application: `streamlit run app.py`

## GDPR Compliance

This project adheres to GDPR principles by:
- Anonymising all personal data
- Implementing appropriate security measures
- Using data only for the specific purpose of wildlife population modelling
- Displaying reduced precision coordinates in visualisations

## Future Development

Potential enhancements include:
- Dynamic coordinate uncertainty based on environmental factors
- Observer identity integration for improved duplicate detection
- Environmental covariates to improve prediction accuracy
- Real-time data collection interface for citizen scientists

## License

This project is available under the MIT License.

MIT License

Copyright (c) 2025 James Fothergill

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Citation

If you use this code or methodology in your research, please cite:

James Fothergill. (2025). An Investigation into the use of Artificial Intelligence to monitor future squirrel populations in Scotland. Teesside University.

## Acknowledgments

- The citizen scientists who contributed observations to the Scottish Squirrel Database
- NBN Atlas for providing access to the dataset
  




  Click on the thumbnail to view the Demonstration video

[![YouTube Video Thumbnail](https://img.youtube.com/vi/HIUu6iCYP1U/0.jpg)](https://www.youtube.com/watch?v=HIUu6iCYP1U)
