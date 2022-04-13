#!/usr/bin/bash
pushd images_to_be_annotated
for f in *.mha; do
	echo Now annotating $f
  export nf="${f%.mha}.overlay.mha"
  ImageViewer -s 0 --fixedSliceDelta 1 --preserveOverlayPaint -W p2,Artery,20,1,p2Needle,7,2 -b 0.5 -S "../annotations/$nf" "$f"
	echo Saving overlay as "../already_annotated_images/$f"
	mv "$f" "../already_annotated_images/"
	echo .
	echo ---------------------------------------------
	echo .
done
popd

echo There are no more images to annotate.
