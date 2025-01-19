import cv2
import numpy as np
import base64
import tempfile
import os
import copy
import pandas as pd
from dash import dash, dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
from flask import send_file
import imageio
from dash import callback_context
import dash_uploader as du
import uuid
from shapely.geometry import MultiPoint
from scipy.stats import gaussian_kde

FONT_AWESOME = (
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css"
)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, FONT_AWESOME])

# Configure the upload folder
UPLOAD_FOLDER_ROOT = tempfile.mkdtemp()
du.configure_upload(app, UPLOAD_FOLDER_ROOT)

# du.upload function (main function)
def get_upload_component(id):
    return du.Upload(
        id=id,
        text='Drag and drop or select a video file',
        max_file_size=10000,  # 10000 MB
        max_files=1,
        filetypes=['mp4', 'avi', 'mov', 'wmv', 'mkv', 'flv'],
        upload_id=uuid.uuid1(),  # Unique session id
    )

# Dropdown options for background subtractors
bg_subtractors = {
    "MOG2": cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=64, detectShadows=True),
    "KNN": cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=800, detectShadows=True),
    "GMG": cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames=120, decisionThreshold=0.8),
    "CNT": cv2.bgsegm.createBackgroundSubtractorCNT(useHistory=True, minPixelStability=1, maxPixelStability=1*60),
    "GSOC": cv2.bgsegm.createBackgroundSubtractorGSOC(nSamples=20, replaceRate=0.01, propagationRate=0.1, hitsThreshold=64),
    "LSBP": cv2.bgsegm.createBackgroundSubtractorLSBP(nSamples=20, LSBPRadius=16, LSBPthreshold=32, Tupper=128, Tlower=8, Tinc=1, Tdec=0.05, noiseRemovalThresholdFacBG=0.001, noiseRemovalThresholdFacFG=0.009)
}

# Define colormap dictionary
colormaps = {
    "Bone": cv2.COLORMAP_BONE,
    "Ocean": cv2.COLORMAP_OCEAN,
    "Pink": cv2.COLORMAP_PINK,
    "Hot": cv2.COLORMAP_HOT
}

# Displaying Hexsticker in the NavBar (Note to display the image logo, just need to create an assets/ and put the logo inside this folder)
# @app.server.route('/AnimalMotionViz.png')
# def serve_hex_png():
#     return send_from_directory(os.path.dirname(__file__), 'AnimalMotionViz.png')

# NavBar
navbar = html.Header(
    [
        dbc.Row([
            dbc.Col(html.Img(src="/assets/AnimalMotionViz.png", height="250px"), width={"size": 1, "offset": 1}),
            dbc.Col(
            [
                html.Span(
                    "AnimalMotionViz: an interactive software tool for tracking and visualizing animal motion patterns using computer vision", 
                    style={
                        'font-family': 'Impact', 
                        'font-weight': '500', 
                        'font-size': '40px', 
                        'color': '#0021A5', 
                        'margin': '10px', 
                        'line-height': '1.4'
                    }
                )
            ],
            width={'size': 9},
            style={'margin': '15px'},
            align='center'
        )
        ])
    ],
    style={
        'background-color': '#FFFFFF', 
        'padding': '10px', 
        'text-align': 'center', 
        'margin': '20px 0px 10px 0px'
    }
)

html.Br(),

# step1: Upload File Section
upload_file = html.Div([
    html.Label(
        'Step 1: Upload input video file', 
        style={
            'font-weight': '700', 
            'font-size': '15px', 
            'margin': '0px'
        }
    ),
    # Video upload component `upload-video`
    html.Div([
        get_upload_component(id='dash-uploader'),
        html.Div(id='upload-video'),
        ],
        style={
            'width': '80%',
            'textAlign': 'center',
            'margin': '20px 20px 20px 50px',
            'font-weight': '700',
            'display': 'inline-block'
        },
    ),
    ],
),

html.Br(),

# step 2: Selecting mask
mask_val = html.Div([
    html.Label(
        [
        'Step 2: Upload a custom mask (optional). This mask image allows tracking motion patterns in a specific region. See the ',
            html.A('doc', href='https://github.com/uf-aiaos/AnimalMotionViz', target='_blank'),
            ' for creating a mask image.'
        ], 
        style={
            'font-weight': '700', 
            'font-size': '15px', 
            'margin': '0px'
        }
    ),
    dcc.Dropdown(
        id='mask-dropdown',
        options=[
            {'label': 'Apply a custom mask', 'value': 'True'},
            {'label': 'Without a mask', 'value': 'False'}
        ],
        value='False',
        style={'margin': '10px'}
    ),
    html.Div(id='upload-mask-container', style={'display': 'none'}, children=[
        dcc.Upload(
            id='upload-mask',
            children=html.Div([
                'Upload a mask image'
            ]),
            style={
                'width': '80%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '20px 20px 20px 50px',
                'font-weight': '700'
            },
            accept='image/*',
            multiple=False
        )
    ]),
]),

html.Br(),

# Step 3: Selecting Background Subtractor Algorithm
background_sub = html.Div([
    html.Label(
        [
        'Step 3: Select a background subtraction algorithm to track animal motion patterns (see the ',
            html.A('doc', href='https://github.com/uf-aiaos/AnimalMotionViz', target='_blank'),
            ' for details of each algorithm). '
        ], 
        style={
            'font-weight': '700', 
            'font-size': '15px', 
            'margin': '0px'
        }
    ),
    dcc.Dropdown(
        id='bg-subtractor-dropdown',
        options=[{'label': key, 'value': key} for key in bg_subtractors.keys()],
        value='MOG2',
        placeholder="Select Background Subtractor",
        style={'margin': '10px'}
    ),
]),

html.Br(),

# Step 4: frequency/step to process the video
freq_size = html.Div([
    html.Label(
        'Step 4: Specify the interval for frame processing (e.g., every nth frame).', 
        style={
            'font-weight': '700', 
            'font-size': '15px', 
            'margin': '0px'
        }
    ),
    dcc.Input(
        id='freq-numeric-input',
        type="number",
        debounce=True,
        required=True,
        value=1,
        min=1,
        style={'margin': '20px', 'width': '98%'}
    ),
]),

html.Br(),

# Step 5: Setting up the kernel size
kernel_size = html.Div([
    html.Label(
        'Step 5: Enter the kernel size for the morphological operation. A small kernel removes minor noise while preserving structures, whereas a larger kernel removes more noise but may also eliminate useful information.', 
        style={
            'font-weight': '700', 
            'font-size': '15px', 
            'margin': '0px'
        }
    ),
    dcc.Input(
        id='ksize-numeric-input',
        type="number",
        debounce=True,
        required=True,
        value=3,
        min=1,
        style={'margin': '20px', 'width': '98%'}
    ),
]),

html.Br(),

# Step 6: Setting up the contours threshold
contours_thresh = html.Div([
    html.Label(
        'Step 6: Set a threshold value to filter the detected motions for calculating the core and full range. A larger threshold removes more small movements and noise.', 
        style={
            'font-weight': '700', 
            'font-size': '15px', 
            'margin': '0px'
        }
    ),
    dcc.Input(
        id='contours-thresh-input',
        type="number",
        debounce=True,
        required=True,
        value=0,
        min=0,
        style={'margin': '20px', 'width': '98%'}
    ),
]),

html.Br(),

# Step 7: Selecting Alpha 
alpha_input = html.Div([
    html.Label(
        [
        'Step 7: Specify the value of the `alpha` (a weight ranging from 0 to 1.0 that controls the contribution of the orginal frame to the final output). See the ',
            html.A('doc', href='https://github.com/uf-aiaos/AnimalMotionViz', target='_blank'),
            ' for details.'
        ], 
        style={
            'font-weight': '700', 
            'font-size': '15px', 
            'margin': '0px'
        }
    ),
    dcc.Input(
        id='alpha-numeric-input',
        type="number",
        debounce=True,
        required=True,
        value=0.9,
        max=1.0,
        min=0.0,
        step=0.1,
        style={'margin': '20px', 'width': '98%'}
    ),
]),

html.Br(),

# Step 8: Selecting Beta 
beta_input = html.Div([
    html.Label(
        [
        'Step 8: Specify the value of the `beta` (a weight ranging from 0 to 1.0 that controls the contribution of the motion patterns to the final output). See the ',
            html.A('doc', href='https://github.com/uf-aiaos/AnimalMotionViz', target='_blank'),
            ' for details.'
        ], 
        style={
            'font-weight': '700', 
            'font-size': '15px', 
            'margin': '0px'
        }
    ),
    dcc.Input(
        id='beta-numeric-input',
        type="number",
        debounce=True,
        required=True,
        value=0.4,
        max=1,
        min=0,
        step=0.1,
        style={'margin': '20px', 'width': '98%'}
    ),
]),

html.Br(),

# Step 9: Selecting colormap
colormap_div = html.Div([
    html.Label(
        'Step 9: Choose a colormap for visualizing the resulting motion patterns.', 
        style={
            'font-weight': '700', 
            'font-size': '15px', 
            'margin': '0px'
        }
    ),
    dcc.Dropdown(
        id='colormap-dropdown',
        options=[
            {'label': key, 'value': key} for key in colormaps.keys()
        ],
        value='Hot',  # Default colormap mode
        placeholder="Select Colormap",
        style={'margin': '10px'}
    ),
]),

html.Br(),

# Step 10: Button to trigger the processing of a video file and display outputs 
button = html.Div([
    html.Label(
        'Step 10: Run the analysis', 
        style={
            'font-weight': '700', 
            'font-size': '15px', 
            'margin': '0px'
        }
    ),
    html.Br(),
    dbc.Button(
        id='process-button', 
        children=[
            html.I(className="fa fa-hourglass-start", style={'margin-right': '5px'}), # add space between icon and text
            "Process the video"
            ], 
        n_clicks=0, 
        color="primary", 
        style={'margin': '20px'}
        ),
    ]
),

html.Br(),

# Download Button for downloading metrics as .csv
download_button = html.Div([
    html.Label(
        'Save the output', 
        style={
            'font-weight': '700', 
            'font-size': '20px', 
            'margin': '0px'
        }
    ),
    html.Br(),
    dbc.Button(
        id='download-button', 
        children=[
            html.I(className="fa fa-download", style={'margin-right': '5px'}), 
            "Download the resulting motion patterns  metrics summary table"
        ], 
        n_clicks=0, 
        color="primary", 
        style={'margin': '20px'}
    ),
    dcc.Download(id="download-metrics"),
    ],
),

# html.Br()

# Define the contact and help section
contact_help = html.Div([
    html.Hr(style={'margin': '10px 0'}),  # Horizontal line with vertical space
    html.Label(
        'Contact information and help:', 
        style={
            'font-weight': '700', 
            'font-size': '20px', 
            'margin': '20px 0px 10px 0px'
        }
    ),
    html.Ul([
        html.Li([
            "Angelo De Castro: ",
            html.A("decastro.a@ufl.edu", href="mailto:decastro.a@ufl.edu", style={'color': '#1a0dab'})
        ]),
        html.Li([
            "Haipeng Yu: ",
            html.A("haipengyu@ufl.edu", href="mailto:haipengyu@ufl.edu", style={'color': '#1a0dab'})
        ])
    ], style={'font-size': '16px', 'font-weight': '500', 'margin': '10px 0px'})
]), 

html.Br(),

# Define the logo and copyright section
lab_logo = html.Div([
    html.Img(src='/assets/AIAOSLab.png', style={'width': '150px', 'margin-top': '40px', 'margin-bottom': '2px'}),  
    html.P(
        'Copyright (C) 2024, code licensed under GPLv3.',
        style={
            'font-size': '15px', 
            'margin': '10px 0px'
        }
    ),
])

# Placeholder for displaying motion heatmap image
heatmap_img = html.Div(
    id='output-heatmap', 
    style={'margin': '10px 0px 10px 0px'}
),

html.Br(),

# Placeholder for displaying core range
core_img = html.Div(
    id='output-core', 
    style={'margin': '10px 0px 10px 0px'}
),

html.Br(),

# Placeholder for displaying the motion heatmap video
processed_vid = html.Div(
    id='output-video', 
    style={'margin': '10px 0px 10px 0px'}
),

html.Br(),

# Metrics Table using AG Grid
getRowStyle={
    "styleConditions": [
        {
            "condition": "params.rowIndex < 8",
            "style": {"backgroundColor": "sandybrown"}  
        },
        {
            "condition": 'params.node.rowIndex >= 8',
            "style": {"backgroundColor": "lightblue"}  
        }
    ]
}

metrics_table = html.Div([
    dag.AgGrid(
        id='metrics-ag-grid',
        columnDefs=[
            {'headerName': 'Metric', 'field': 'Metric', 'width': 290},
            {'headerName': 'Description', 'field': 'Description', 'width': 645},
            {'headerName': 'Value', 'field': 'Value', 'width': 220},
        ],
        rowData=[],                         # Initially empty, will be updated in the callback
        columnSize="autoSizeAllColumns",
        getRowStyle=getRowStyle,
        csvExportParams={'filename': 'metrics.csv'},
        style={
            'height': '450px', 
            'width': '98%', 
            'margin': '10px 0px 10px 0px'
        },
    ),
]),

footnote = html.Div(
    [
        "The rows highlighted in orange represent results from the motion heatmap image on the left, while the rows highlighted in light blue represent results from the core and full range image on the right."
    ], 
    style={
        'textAlign': 'left',
        'fontSize': '16px',
        'marginTop': '10px'
    }
)


html.Br(),

# Callback function that when mask is true, dcc.Upload will display, else, default
@app.callback(
    Output('upload-mask-container', 'style'),
    [Input('mask-dropdown', 'value')]
)
def update_mask_upload_visibility(mask_value):
    if mask_value == 'True':
        return {'display': 'block'}
    else:
        return {'display': 'none'} 
    
# Function that finds the top 3 peak intensity locs
def find_peak_locations(heatmap, num_peaks):
    # Find indices of the top num_peaks intensity values
    indices = np.unravel_index(np.argsort(heatmap.ravel())[-num_peaks:], heatmap.shape)
    # Convert indices to coordinates
    coordinates = list(zip(indices[1], indices[0]))  # Swap x and y due to row-column indexing
    return coordinates

# Function that draw a marker in the image representing the peak intensity locs
def draw_peak_markers(image, peak_locations, 
                      shapes=['circle', 'square', 'triangle'], 
                      colors=[(0, 20, 20), (0, 0, 255), (255, 0, 0)], 
                      radius=10, 
                      thickness=2
):

    # Draw markers on the image
    for i, (loc, shape, color) in enumerate(zip(peak_locations, shapes, colors)):
        if shape == 'circle':
            cv2.circle(image, loc, radius, color, thickness)
        elif shape == 'square':
            x, y = loc
            cv2.rectangle(image, (x - radius, y - radius), (x + radius, y + radius), color, thickness)
        elif shape == 'triangle':
            x, y = loc
            points = np.array([[x, y - radius], [x - int(radius * 0.86), y + int(radius * 0.5)], [x + int(radius * 0.86), y + int(radius * 0.5)]], np.int32)
            points = points.reshape((-1, 1, 2))
            cv2.polylines(image, [points], True, color, thickness)

# Function that caculate the percentage of used region in each of the 4 quadrants            
def calculate_quadrant_percentage(heatmap):
    # Determine the dimensions of the image
    height, width = heatmap.shape[:2]  
    print('size: ', height, width)
    # Divide the image into four quadrants
    quadrant1 = heatmap[0:height//2, width//2:]
    quadrant2 = heatmap[0:height//2, 0:width//2]
    quadrant3 = heatmap[height//2:, 0:width//2]
    quadrant4 = heatmap[height//2:, width//2:]

    # Calculate percentage of used region for each quadrant
    # Top right quadrant
    q1_percentage = (cv2.countNonZero(quadrant1) / (quadrant1.shape[0] * quadrant1.shape[1])) * 100
    q1_percentage_rd = round(q1_percentage, 2)
    q1_percentage_str = f"{q1_percentage_rd}%"
    print('q1 used pixels: ', cv2.countNonZero(quadrant1))
    print('q1_size: ', quadrant1.shape[0] * quadrant1.shape[1])
    
    # Top left quadrant
    q2_percentage = (cv2.countNonZero(quadrant2) / (quadrant2.shape[0] * quadrant2.shape[1])) * 100
    q2_percentage_rd = round(q2_percentage, 2)
    q2_percentage_str = f"{q2_percentage_rd}%"
    print('q2 used pixels: ', cv2.countNonZero(quadrant2))
    print('q2_size: ', quadrant2.shape[0] * quadrant2.shape[1])
    
    # Bottom left quadrant
    q3_percentage = (cv2.countNonZero(quadrant3) / (quadrant3.shape[0] * quadrant3.shape[1])) * 100
    q3_percentage_rd = round(q3_percentage, 2)
    q3_percentage_str = f"{q3_percentage_rd}%"
    print('q3 used pixels: ', cv2.countNonZero(quadrant3))
    print('q3_size: ', quadrant3.shape[0] * quadrant3.shape[1])
    
    # Bottom right quadrant
    q4_percentage = (cv2.countNonZero(quadrant4) / (quadrant4.shape[0] * quadrant4.shape[1])) * 100
    q4_percentage_rd = round(q4_percentage, 2)
    q4_percentage_str = f"{q4_percentage_rd}%"
    print('q4 used pixels: ', cv2.countNonZero(quadrant4))
    print('q4_size: ', quadrant4.shape[0] * quadrant4.shape[1])
    
    return q1_percentage_str, q2_percentage_str, q3_percentage_str, q4_percentage_str

# function to visualize core and full range
def overlay_core_on_frame(first_frame, full_positions, core_positions):
    # Create a copy of the first frame to draw on
    frame_copy = first_frame.copy()

    # Draw all detected positions as small blue dots
    for pos in full_positions:
        cv2.circle(frame_copy, (int(pos[0]), int(pos[1])), radius=3, color=(255, 0, 0), thickness=-1)

    # Highlight core range positions as larger red dots
    for pos in core_positions:
        cv2.circle(frame_copy, (int(pos[0]), int(pos[1])), radius=6, color=(0, 0, 255), thickness=-1)

    # draw a convex hull around all positions
    if len(full_positions) > 2:
        hull = MultiPoint(full_positions).convex_hull
        if not hull.is_empty:
            points = np.array(hull.exterior.coords, dtype=np.int32)
            cv2.polylines(frame_copy, [points], isClosed=True, color=(0, 255, 255), thickness=2)

    return frame_copy

# Processing function
def process_video(contents, 
                  mask_contents, 
                  bg_subtractor_name='MOG2', 
                  freq=1,
                  k_size=3,
                  thresh_size=0,
                  alpha=0.9, 
                  beta=0.4, 
                  colormap_name='Hot'
):
    print("Processing video...")
        
    # For Mask (if mask is uploaded)    
    mask_image = None
    if mask_contents:
        mask_content_type, mask_content_string = mask_contents.split(',')
        decoded_mask = base64.b64decode(mask_content_string)
        nparr = np.frombuffer(decoded_mask, np.uint8)
        mask_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # variables for video frames
    capture = cv2.VideoCapture(contents)
    length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)  # I added this to get the fps in the input video which will be used to convert the processed images to output video; otherwise the converted output video may have different legnth than the input
    print(f"Total frames: {length}, FPS: {fps}") # print out the frames and fps of input video

    # Obtaining bgsegm from the bgsegm dictionary
    background_subtractor = bg_subtractors.get(bg_subtractor_name)
    
    # Obtaining colormap from the colormap dictionary
    mode = colormaps.get(colormap_name)
    
    # other variables
    first_iteration_indicator = 1
    processed_frames = []

    # Positions to track centroids
    positions = []

    # Metrics
    metrics = {
        'Peak Intensity Locations': [],
        'Overall Percentage of Used Region': [],
        'Quadrant 1 Percentage of Used Region': [],
        'Quadrant 2 Percentage of Used Region': [],
        'Quadrant 3 Percentage of Used Region': [],
        'Quadrant 4 Percentage of Used Region': [],
        'Full Range Convex Hull Area': []
    }
    
    # Process all frames of the video
    for i in range(0, length, freq):
        ret, frame = capture.read()
        if not ret:
            break
        
        # If first frame
        if first_iteration_indicator == 1:
            # Get the first frame of the video
            first_frame = copy.deepcopy(frame)
            height, width = frame.shape[:2]
            accum_motion = np.zeros((height, width), np.uint8) 
            first_iteration_indicator = 0
            
        else:
            # Apply background subtraction
            xfilter = background_subtractor.apply(frame)
            
            # To remove small amounts of movements, such as the wind, a small bird flying
            threshold = 200
            maxValue = 1
            ret, th1 = cv2.threshold(xfilter, threshold, maxValue, cv2.THRESH_BINARY)

            # Create an elliptical structuring element for morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))

            # Apply morphological opening to remove small noise from the image using the elliptical kernel
            th1_morph = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kernel)
            
            # Apply mask if provided
            if mask_image is not None:
                gray_mask = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray_mask, 0, 255, cv2.THRESH_BINARY)
                th1_mask = cv2.bitwise_and(th1_morph, th1_morph, mask=mask) 
            else: 
                th1_mask = th1_morph     

            # Accumulate motion over time 
            accum_motion = cv2.add(accum_motion, th1_mask)

            # Find contours to locate movement
            contours, _ = cv2.findContours(th1_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) > thresh_size:  # Filter noise
                    M = cv2.moments(contour)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        positions.append([cx, cy])
            
            # Create motion heatmap
            color_image_video = cv2.applyColorMap(accum_motion, mode)
            video_frame = cv2.addWeighted(frame, alpha, color_image_video, beta, 0)
            processed_frames.append(video_frame)
    
    #Generate heatmap with the first frame as background
    color_image = cv2.applyColorMap(accum_motion, mode)
    result_overlay = cv2.addWeighted(first_frame, alpha, color_image, beta, 0)        
            
    # Obtain heatmap from the processed frame
    heatmap = copy.deepcopy(accum_motion)

    # Core and Full Range Calculation
    if positions:
        positions_np = np.array(positions)

        # Kernel Density Estimation
        kde = gaussian_kde(positions_np.T)
        densities = kde(positions_np.T)

        # Find the 50% densest area
        density_threshold = np.percentile(densities, 50)
        core_positions = positions_np[densities >= density_threshold]
        print(f"Core Range (50% densest positions): {core_positions}")

        # Sort points by density and select 95% isopleth
        full_threshold = np.percentile(densities, 5)  # Exclude the lowest 5% of densities
        full_positions = positions_np[densities > full_threshold]
        print(f"Full Range (95% densest positions): {full_positions}")

        core_points = MultiPoint(core_positions)
        convex_core = core_points.convex_hull
        full_points = MultiPoint(full_positions)
        convex_full = full_points.convex_hull

        # Visualizing core and full range
        core_image = overlay_core_on_frame(first_frame, full_positions, core_positions)
        
        core_range = convex_core.area
        full_range = convex_full.area
        print(f"Core Range Convex Hull Area: {core_range}")
        print(f"Full Range Convex Hull Area: {full_range}")
    else:
        print("No positions detected for Core Range and Full Range calculations.")

    # Find top three peak intensity locations in the heatmap
    location = find_peak_locations(heatmap, 3)

    # Draw circles or stars on the top three peak intensity locations
    draw_peak_markers(result_overlay, 
                      location, 
                      shapes=['circle', 'square', 'triangle'], 
                      colors=[(0, 20, 20), (0, 0, 255), (255, 0, 0)], 
                      radius=10, 
                      thickness=2
    )

    # Calculate the overall percentage of the used region
    used_pixels_count = cv2.countNonZero(heatmap)
    print('overall used pixels: ', used_pixels_count)   # for verification purposes

    # Calculate the total number of pixels in the accum_image
    total_pixels = heatmap.shape[0] * heatmap.shape[1]
    print('overall size: ', total_pixels)   # for verification purposes

    # Calculate the percentage of usage
    used_region_percentage = (used_pixels_count / total_pixels) * 100
    used_region_percentage_rd = round(used_region_percentage, 2)
    
    used_region_percentage_str = f"{used_region_percentage_rd}%"   # Convert float to string for display (percent sign)
    
    # For displaying the percentage of used region in each of the 4 quadrants
    q1_percentage_str, q2_percentage_str, q3_percentage_str, q4_percentage_str = calculate_quadrant_percentage(heatmap)
    
    # Displaying metrics
    metrics['Peak Intensity Locations'] = location
    metrics['Overall Percentage of Used Region'] = used_region_percentage_str
    metrics['Quadrant 1 Percentage of Used Region'] = q1_percentage_str
    metrics['Quadrant 2 Percentage of Used Region'] = q2_percentage_str
    metrics['Quadrant 3 Percentage of Used Region'] = q3_percentage_str
    metrics['Quadrant 4 Percentage of Used Region'] = q4_percentage_str
    metrics['Core Range Convex Hull Area'] = core_range
    metrics['Full Range Convex Hull Area'] = full_range
            
    capture.release()
    cv2.destroyAllWindows()
    
    print("Video processing completed.")
    return result_overlay, processed_frames, core_image, metrics, fps, height, width # return the fps for converting the processed frames to output video; this will ensure the output and input videos have same length in time 



# Main Callback Decorator and Callback Function
@app.callback(
    Output('output-heatmap', 'children'),
    Output('output-core', 'children'),
    Output('output-video', 'children'),
    Output('metrics-ag-grid', 'rowData'),
    [Input('dash-uploader', 'isCompleted'),
     Input('process-button', 'n_clicks'),
     Input('upload-mask', 'contents')],
    [State('dash-uploader', 'fileNames'),
     State('dash-uploader', 'upload_id'),
     State('bg-subtractor-dropdown', 'value'),
     State('freq-numeric-input', 'value'),
     State('ksize-numeric-input', 'value'),
     State('contours-thresh-input', 'value'),
     State('alpha-numeric-input', 'value'),
     State('beta-numeric-input', 'value'),
     State('colormap-dropdown', 'value')],
    prevent_initial_call=True,
    running=[(Output("process-button", "disabled"), True, False)]    # disabling the button when the callback is running
)
def update_processed_video(is_completed, 
                           n_clicks, 
                           mask_contents,
                           file_names,
                           upload_id,
                           bg_subtractor_name, 
                           freq,
                           k_size,
                           thresh_size,
                           alpha, 
                           beta, 
                           colormap_name
):
    print("Updating processed video...")
    # Determine which input triggered the callback; 
    # here `button_id` is used to trigger the callback only when `process-button` is clicked which can prevent the program automatically running when update input video or mask. 
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    # if contents is not None and n_clicks:
    if is_completed is not None and button_id == 'process-button':
        # callback for the du.upload component (initiating the function)
        upload_folder = os.path.join(UPLOAD_FOLDER_ROOT, upload_id)
        for file_name in file_names:
            file_path = os.path.join(upload_folder, file_name)
            result_overlay, processed_frames, core_image, metrics, fps, height, width = process_video(file_path, 
                                                                    mask_contents, 
                                                                    bg_subtractor_name,
                                                                    freq,
                                                                    k_size,
                                                                    thresh_size,
                                                                    alpha, 
                                                                    beta, 
                                                                    colormap_name
                                                                    )
            
            if result_overlay is not None and core_image is not None and processed_frames is not None:
                # Save the last frame with overlay to a temporary file
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                    temp_file_path = temp_file.name
                    cv2.imwrite(temp_file_path, result_overlay)   
                
                # Read the last frame with overlay as binary data
                with open(temp_file_path, 'rb') as file:
                    img_data = file.read()    

                # Encode the image data to base64 for displaying in the app
                img_base64 = base64.b64encode(img_data).decode('utf-8')
                
                # Remove the temporary file
                os.remove(temp_file_path)

                # for the core and full range visualization
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                    temp_file_path2 = temp_file.name
                    cv2.imwrite(temp_file_path2, core_image)

                with open(temp_file_path2, 'rb') as file2:
                    img_data2 = file2.read()

                img_base64_2 = base64.b64encode(img_data2).decode('utf-8')
                os.remove(temp_file_path2)
                
                # html tag for the heatmap image
                motionheatmap_image = dcc.Graph(
                    id="heatmap-img-graph",
                    figure={
                        "data": [],
                        "layout": {
                            "autosize": True,
                            "width": width/2,
                            "height": height/2,
                            "plot_bgcolor": "#FFF",
                            "paper_bgcolor": "#FFF",
                            "margin": {"l": 5, "b": 5, "t": 5, "r": 5}, 
                            "dragmode": "select", 
                            "xaxis": {
                                "range": (0, width),
                                "tickwidth": 1,
                                "scaleratio": 1,
                                "scaleanchor": "y",
                                "color": "white",
                                "gridcolor": "#FFF",
                            },
                            "yaxis": {
                                "range": (0, height),
                                "tickwidth": 1,
                                "color": "white",
                                "gridcolor": "#FFF",
                            },
                            "images": [
                                {
                                    "source": f"data:image/jpg;base64,{img_base64}",
                                    "x": 0,
                                    "y": 0,
                                    "sizex": width,
                                    "sizey": height,
                                    "xref": "x",
                                    "yref": "y",
                                    "yanchor": "bottom",
                                    "sizing": "contain",
                                    "layer": "above",
                                }
                            ],
                        },
                    },
                )

                # html tag for the core and full range image
                core_image = dcc.Graph(
                    id="core-range-graph",
                    figure={
                        "data": [],
                        "layout": {
                            "autosize": True,
                            "width": width/2,
                            "height": height/2,
                            "plot_bgcolor": "#FFF",
                            "paper_bgcolor": "#FFF",
                            "margin": {"l": 5, "b": 5, "t": 5, "r": 5}, 
                            "dragmode": "select", 
                            "xaxis": {
                                "range": (0, width),
                                "tickwidth": 1,
                                "scaleratio": 1,
                                "scaleanchor": "y",
                                "color": "white",
                                "gridcolor": "#FFF",
                            },
                            "yaxis": {
                                "range": (0, height),
                                "tickwidth": 1,
                                "color": "white",
                                "gridcolor": "#FFF",
                            },
                            "images": [
                                {
                                    "source": f"data:image/jpg;base64,{img_base64_2}",
                                    "x": 0,
                                    "y": 0,
                                    "sizex": width,
                                    "sizey": height,
                                    "xref": "x",
                                    "yref": "y",
                                    "yanchor": "bottom",
                                    "sizing": "contain",
                                    "layer": "above",
                                }
                            ],
                        },
                    },
                )

                # Create a temporary file with a '.mp4' extension
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                    temp_file_path = temp_file.name

                    # Initialize the imageio writer
                    writer = imageio.get_writer(temp_file_path, fps=fps, codec='libx264', format='FFMPEG', pixelformat='yuv420p')

                    # Print out video converting progress: 10 times
                    num_frames = len(processed_frames)
                    interval = max(1, num_frames // 10)  # Calculate interval for printing 10 lines
        
                    # Write each processed frame into the video
                    for i, frame in enumerate(processed_frames):
                        writer.append_data(frame[..., ::-1])  # Convert from BGR to RGB as imageio expects RGB format
                        # Only print progress at the calculated intervals
                        if (i + 1) % interval == 0 or (i + 1) == num_frames:
                            print(f"Converted {i + 1} frames out of {num_frames} to output motion video")

                    # Close the writer to finalize the video file
                    writer.close()
                            
                # Serve the video using Flask
                video_url = f"/video_feed/{os.path.basename(temp_file_path)}"

                # html tag for the heatmap video
                video_tag = html.Video(
                    src=video_url, 
                    controls=True, 
                    style={'width': '98%'}, 
                    autoPlay=False, 
                    loop=True, 
                    title='motion-heatmap-video',
                    preload='auto'  # Enable progressive loading
                )

                # Fixed descriptions for the top three largest peak locations
                fixed_descriptions = ['The point with the largest movements, marked by a black circle shape.', 
                                    'The point with the second largest movements, marked by a red square shape.', 
                                    'The point with the third largest movements, marked by a blue triangle shape.']
                
                # Prepare metrics data for the AG Grid
                peak_locations_data = [{'Metric': 'Peak Intensity Location', 'Description': fixed_descriptions[i], 'Value': str(location)} for i, location in enumerate(metrics['Peak Intensity Locations'])]
                peak_locations_df = pd.DataFrame(peak_locations_data)

                # Create a DataFrame for overall percentage
                overall_percentage_df = pd.DataFrame(
                    {
                        'Metric': ['Overall Percentage of Used Region'], 
                        'Description': 'The ratio of movement area to the total available space.',
                        'Value': [metrics['Overall Percentage of Used Region']] 
                    }
                )
                
                # Dataframe for percentage in the quadrants
                q1_percentage_df = pd.DataFrame(
                    {
                        'Metric': ['Quadrant 1 Percentage of Used Region'], 
                        'Description': 'The ratio of movement area to the total available space in top-right quadrant.',
                        'Value': [metrics['Quadrant 1 Percentage of Used Region']]
                    }
                )

                q2_percentage_df = pd.DataFrame(
                    {
                        'Metric': ['Quadrant 2 Percentage of Used Region'], 
                        'Description': 'The ratio of movement area to the total available space in top-left quadrant.',
                        'Value': [metrics['Quadrant 2 Percentage of Used Region']]
                    }
                )

                q3_percentage_df = pd.DataFrame(
                    {
                        'Metric': ['Quadrant 3 Percentage of Used Region'], 
                        'Description': 'The ratio of movement area to the total available space in bottom-left quadrant.',
                        'Value': [metrics['Quadrant 3 Percentage of Used Region']]
                    }
                )

                q4_percentage_df = pd.DataFrame(
                    {
                        'Metric': ['Quadrant 4 Percentage of Used Region'], 
                        'Description': 'The ratio of movement area to the total available space in bottom-right quadrant.',
                        'Value': [metrics['Quadrant 4 Percentage of Used Region']]
                    }
                )

                # dataframe for core and full range metrics
                core_range_df = pd.DataFrame(
                    {
                        'Metric': ['Core Range (50% Isopleth)'], 
                        'Description': 'The red region highlights the most intensely used 50% of the total area where animals frequently moved or stayed.',
                        'Value': [f"{metrics['Core Range Convex Hull Area']} pixels"]
                    }
                )

                full_range_df = pd.DataFrame(
                    {
                        'Metric': ['Full Range (95% Isopleth)'], 
                        'Description': 'The region within the yellow line represents the smallest convex polygon that encloses 95% of the total movement area, including the red and blue regions.',
                        'Value': [f"{metrics['Full Range Convex Hull Area']} pixels"]
                    }
                )

                # Concatenate DataFrames for peak intensity locations and overall percentage
                combined_df = pd.concat(
                    [
                        peak_locations_df, 
                        overall_percentage_df, 
                        q1_percentage_df, 
                        q2_percentage_df, 
                        q3_percentage_df, 
                        q4_percentage_df,
                        core_range_df,
                        full_range_df
                    ], 
                    ignore_index=True
                )

                #Convert DataFrame to a list of dictionaries (rows) for the AG Grid
                row_data = combined_df[['Metric', 'Value', 'Description']].to_dict('records')
                
                print("Processed video ready for display.")
                # Display the last frame with overlay, motion heatmap video, and AG Grid table in the app
                return motionheatmap_image, core_image, video_tag, row_data

    return html.Div(), html.Div(), html.Div(), []  # Return empty divs if no output to display

# Flask route for serving the video file
@app.server.route('/video_feed/<video_filename>')
def serve_video(video_filename):
    video_path = os.path.join(tempfile.gettempdir(), video_filename)
    
    # Send video as a file response
    return send_file(video_path)

# Callback to download the AG Grid data to CSV
@app.callback(
    Output('download-metrics', 'data'),
    [Input('download-button', 'n_clicks')],
    [State('metrics-ag-grid', 'rowData')],
    prevent_initial_call=True
)
def download_csv(n_clicks, row_data):
    if n_clicks:
        if row_data:
            df = pd.DataFrame(row_data)
            return dcc.send_data_frame(df.to_csv, "metrics.csv")
        else:
            return "No data to download."

# Main App Layout
app.layout = html.Div([
    dbc.Row(navbar),
    dbc.Row([
        dbc.Col([
            html.Label(
                "Configure video processing parameters:", 
                style={
                    'font-family': 'Impact', 
                    'font-weight': '500', 
                    'font-size': '24px', 
                    'margin': '0px 0px 30px 0px', 
                    'color': '#0021A5'
                }
            ),
            dbc.Row(upload_file), 
            dbc.Row(mask_val), 
            dbc.Row(background_sub),
            dbc.Row(freq_size),
            dbc.Row(kernel_size),
            dbc.Row(contours_thresh),
            dbc.Row(alpha_input),
            dbc.Row(beta_input),
            dbc.Row(colormap_div),
            dbc.Row(button), 
            dbc.Row(download_button),
            dbc.Row(contact_help),
            dbc.Row(lab_logo)
        ],
        width=3,
        style={
            "height": 1800, 
            "width": 700,
            "border": "5px solid #0021A5",
            "marginTop": 40,
            "marginBottom": 20,
            "marginLeft": 40,
            "marginRight": 30,
            "padding": 20
            }
        ),
        dbc.Col([
            html.Label(
                "Results: ", 
                style={
                    'font-family': 'Impact', 
                    'font-weight': '500', 
                    'font-size': '24px', 
                    'margin': '40px 0px 10px 0px', 
                    'color': '#0021A5'
                }
            ),
            dbc.Spinner(children=[
                dbc.Row(
                    dbc.Tabs(
                        [
                            dbc.Tab(
                                dbc.Row([
                                    dbc.Col(
                                        heatmap_img,
                                        width=6,
                                        style={'padding': '10px'},
                                    ),
                                    dbc.Col(
                                        core_img,
                                        width=6,
                                        style={'padding': '10px'},
                                    ),
                                    ]
                                ), 
                                label='Motion patterns in image',
                                label_style={
                                    'font-family': 'Impact', 
                                    'font-weight': '500', 
                                    'font-size': '20px', 
                                    'margin': '0px', 
                                    'color': '#0021A5'
                                },
                                active_label_style={
                                    'font-family': 'Impact', 
                                    'font-weight': '500', 
                                    'font-size': '20px', 
                                    'margin': '0px', 
                                    'color': '#FA4616'
                                },
                                tab_style={
                                    'width': 300, 
                                    'text-align': 'center'
                                }
                            ),
                            dbc.Tab(
                                processed_vid, 
                                label='Motion patterns in video',
                                label_style={
                                    'font-family': 'Impact', 
                                    'font-weight': '500', 
                                    'font-size': '20px', 
                                    'margin': '0px', 
                                    'color': '#0021A5'
                                },
                                active_label_style={
                                    'font-family': 'Impact', 
                                    'font-weight': '500', 
                                    'font-size': '20px', 
                                    'margin': '0px', 
                                    'color': '#FA4616'
                                },
                                tab_style={
                                    'width': 300, 
                                    'text-align': 'center'
                                }
                            ),
                        ]
                    ),
                ),
                dbc.Row(metrics_table),
                dbc.Row(footnote)
            ],
            color='#0021A5',
            type='border',
            fullscreen=False,
            delay_hide=150,
            spinner_style={
                'margin': '450px 0px 0px 0px', 
                'width': '7rem', 
                'height': '7rem'
            }
            ),
        ],  
        )
    ],
    style={'margin': '0px 0px 40px 0px'} 
    )
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_ui=True, dev_tools_props_check=False, host='127.0.0.1', port=8050)
