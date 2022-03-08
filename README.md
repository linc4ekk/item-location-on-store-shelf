# Item-location-on-a-store-shelf
Goal: find locations of a given item on a store shelf. 

Legend: there is a request from the international retail group that wants a CV system to verify the layout of products on its shelves. Some brands are buying out the top spots on shelves for their goods, so it's crucial to have them at the right places. You will develop a system to control merchandiser work in this assignment. 

You both agreed that it's enough to write a simple function, computing item locations for a demo project. This function should get two images.

The "Gallery" image is the same as above (only without red boxes around juice).

"Query" image will contain one item cropped to its margins.

 

The output should be an array of locations [(x_min, y_min, width, height), ...].

Values are in the scale 0-1 (yolo format). (x_min, y_min) is left-upper corner of a bounding box, x_min = x_min *in pixels* / image_width.
