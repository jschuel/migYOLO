from migYOLO.utils.readYAML import read_config_file
import xml.etree.ElementTree as ET
import cmasher as cmr
import matplotlib.colors as mcolors
import numpy as np

def create_xml(master_config_path, xml_output_path):
    master_config = read_config_file(master_config_path)
    
    objects = master_config['Objects']
    
    # Get enabled objects
    enabled_objects = [label for label, enabled in objects.items() if enabled]
    
    # Generate colors from colormaps
    object_colors = cmr.get_sub_cmap('viridis', 1, 0.2, N=len(enabled_objects))
    object_colors_hex = [mcolors.to_hex(color) for color in object_colors.colors]
        
    # Create root element
    root = ET.Element("View")
    
    # Create Image element
    image = ET.SubElement(root, "Image", name="image", value="$image", zoom="true")
    
    # Create RectangleLabels element
    rect_labels = ET.SubElement(root, "RectangleLabels", name="bbox", toName="image")
    
    # Add labels to RectangleLabels
    for i, label in enumerate(enabled_objects):
        ET.SubElement(rect_labels, "Label", value=label, background=object_colors_hex[i])
    
    # Convert the XML tree to a string
    xml_string = ET.tostring(root, encoding='unicode')

    # Write the XML string to a file, omitting the XML declaration
    with open(xml_output_path, 'w') as xml_file:
        xml_file.write(xml_string)

if __name__ == '__main__':
    master_config_path = 'labelStudio_configuration.yaml'
    xml_output_path = 'output.xml'
    
    create_xml(master_config_path, xml_output_path)
    print(f"XML configuration saved to {xml_output_path}")
