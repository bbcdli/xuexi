#List of content
#rename_sequence.sh #create_vid_from_images.sh
#make_vid_using_images
#use parameter on command line

######################################################################
#rename_sequence.sh change set of images with sequence name and fixed digit length
a=1
for i in *.png; do
  new=$(printf "fr_%04d.png" "$a") #04 pad to length of 4
  echo $new
  mv -i -- "$i" "$new"
  let a=a+1
done

######################################################################
#make_vid_using_images.sh
#ffmpeg -r 1/5 -f image2 -i fr_%05d.png a.mp4 #-r set fps (hold sec per image)


######################################################################
#use parameter on command line, $1 stands for the first parameter when entering sh file in command line, inside sh file the parameter is recognized as $1
e.g: ./run_.sh train, train is the first parameter
if [[ $1 == 'train' ]]; then
  #do sth
fi
######################################################################




######################################################################




######################################################################



######################################################################



######################################################################
