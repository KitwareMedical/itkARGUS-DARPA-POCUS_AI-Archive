::===============================================================
:: This batch script automates an annotation workflow using 
:: ImageViewer to annotate a set of images. Images in the directory 
:: "images_to_be_annotated" are taken as input. Once the annotation 
:: is complete, closing ImageViewer causes the input file and saved 
:: overlay to move to the "already_annotated_images_and_overlays"
:: directory.
::===============================================================

@ECHO OFF

TITLE Command Prompt for ImageViewer Annotations
ECHO This CMD is automating a workflow for annotating images with ImageViewer.
ECHO Close this window to stop working on annotations. Your progress will be saved.
ECHO Input images are located in "images_to_be_annotated".
ECHO Output overlays and annotated images will be moved to "already_annotated_images_and_overlays".
ECHO Use [ and ] to change paint radius.
ECHO Hold SHIFT to erase paint.
ECHO Press "." to advance the slice, and "," to go back a slice.
ECHO Press SPACEBAR to advance to the next workflow.
ECHO The workflow order: High Confidence Artery, Low Confidence Artery, High Confidence Needle, Low Confidence Needle.
ECHO.  

PAUSE

for %%f in (.\images_to_be_annotated\*) do (
	echo Now annotating %%f
	.\ImageViewer\ImageViewer.exe -s 0 --fixedSliceDelta 3 --preserveOverlayPaint -w p,HiConfArtery,10,1,p,LoConfArtery,10,2,p,HiConfNeedle,5,3,p,LoConfNeedle,5,4 -b 0.5 -S ".\annotations\%%~nf" "%%f"
	echo Saving overlays with prefix .\annotations\%%~nf
	move "%%f" ".\already_annotated_images\%%~nf%%~xf"
	echo.
	echo ---------------------------------------------
	echo.
)

ECHO There are no more images to annotate.

PAUSE
