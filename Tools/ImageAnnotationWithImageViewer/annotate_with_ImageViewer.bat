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
ECHO Use { and } to change paint color. Use [ and ] to change paint radius.
ECHO Your current paint color and radius are in the bottom right corner. Color 0 erases.
ECHO.  

PAUSE

for %%f in (.\images_to_be_annotated\*) do (
	echo Now annotating %%f
	.\ImageViewer\ImageViewer.exe -M Paint -S .\already_annotated_images_and_overlays\%%~nf_overlay.mha %%f
	echo Saving overlay as .\already_annotated_images_and_overlays\%%~nf_overlay.mha
	move %%f .\already_annotated_images_and_overlays\%%~nf%%~xf
	echo.
	echo ---------------------------------------------
	echo.
)

ECHO There are no more images to annotate.

PAUSE
