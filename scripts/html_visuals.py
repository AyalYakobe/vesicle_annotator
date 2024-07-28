import os


def generate_html(image_paths, save_dir):
    # Sort images by prediction type
    dv_images = []
    cv_images = []
    dvh_images = []

    for image_path in image_paths:
        if 'DV' in image_path:
            dv_images.append(image_path)
        elif 'CV' in image_path:
            cv_images.append(image_path)
        else:
            dvh_images.append(image_path)

    # Generate HTML content
    html_content = '<html><body>\n'

    # Add sections for each prediction type
    if dv_images:
        html_content += '<h2>DV Predictions</h2>\n'
        for image_path in dv_images:
            relative_path = os.path.relpath(image_path, save_dir)
            html_content += f'<img src="{relative_path}" alt="DV Prediction Image" style="width:100%;">\n'

    if cv_images:
        html_content += '<h2>CV Predictions</h2>\n'
        for image_path in cv_images:
            relative_path = os.path.relpath(image_path, save_dir)
            html_content += f'<img src="{relative_path}" alt="CV Prediction Image" style="width:100%;">\n'

    if dvh_images:
        html_content += '<h2>DVH Predictions</h2>\n'
        for image_path in dvh_images:
            relative_path = os.path.relpath(image_path, save_dir)
            html_content += f'<img src="{relative_path}" alt="DVH Prediction Image" style="width:100%;">\n'

    html_content += '</body></html>'

    html_path = os.path.join(save_dir, 'index.html')
    with open(html_path, 'w') as html_file:
        html_file.write(html_content)
    print(f"Saved HTML file to {html_path}")