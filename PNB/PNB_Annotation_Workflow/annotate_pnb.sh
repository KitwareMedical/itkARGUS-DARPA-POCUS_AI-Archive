#!/usr/bin/bash
pushd images_to_be_annotated
for f in *.mha; do
	echo Now annotating $f
  export nf="${f%.mha}.overlay.mha"
  ImageViewer.exe -s 0 --fixedSliceDelta 3 --preserveOverlayPaint -w p,HiConfArtery,20,1,p,LoConfArtery,20,2,p,HiConfNeedle,7,3,p,LoConfNeedle,7,4 -b 0.5 -S "../annotations/$nf" "$f"
	echo Saving overlay as "../already_annotated_images/$f"
	mv "$f" "../already_annotated_images/"
	echo .
	echo ---------------------------------------------
	echo .
done
popd

echo There are no more images to annotate.
