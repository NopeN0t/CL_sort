# Usage
Note : Add this program to global path variable Before using <br>
There is two opration mode for this program <br>

Sample based <br>
 - The program will detect files inside a sub folder and use it as sample for sorting <br> 

Cluster Mode <br>
 - The program will automaticly switch to clustering mode when there is no sample files <br>
<br>
Commands

````
 CL_sort "folder path"
````

Arguments <br>
-l or --lang 				add other language for extraction <br>
-m or --max_page 		total amount of pages can be extract from pdf <br>
-t or --text_max 		total amount of text can be extract from each file <br>
-d or --doc_img 		Allow program to extract data from image inside doc files <br>
-p or --show_path 	print file path into console while running <br>
-i or --image_size 	size of image scaled to when reading them <br>
-e or --encoder 		encoder used to encode text <br>
-s or --stage       set stage of opration of the program <br>

## Dependencies
 - odfpy
 - pillow
 - pandas
 - pytesseract
 - python-docx
 - PyMuPDF
 - tqdm
 - tensorflow
 - tensorflow-hub
 - tensorflow-text
 - scikit-learn
