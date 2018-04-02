#rename_sequence.sh #create_vid_from_images.sh




#rename_sequence.sh change set of images with sequence name and fixed digit length
a=1
for i in *.png; do
  new=$(printf "fr_%04d.png" "$a") #04 pad to length of 4
  echo $new
  mv -i -- "$i" "$new"
  let a=a+1
done

#create_vid_from_images.sh
#ffmpeg -r 1/5 -f image2 -i fr_%05d.png a.mp4 #-r set fps (hold sec per image)


