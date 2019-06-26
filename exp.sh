#sh run.sh 2 10 16 2     _timeline/resnext resnext
#sh run.sh 2 10 16 4     _timeline/resnext resnext
#sh run.sh 2 10 16 8     _timeline/resnext resnext
#sh run.sh 2 10 16 16    _timeline/resnext resnext

#sh run.sh 2 10 32 2     _timeline/resnext resnext
#sh run.sh 2 10 32 4     _timeline/resnext resnext
#sh run.sh 2 10 32 8     _timeline/resnext resnext
#sh run.sh 2 10 32 16    _timeline/resnext resnext

#sh run.sh 2 10 64 2     _timeline/resnext resnext
#sh run.sh 2 10 64 4     _timeline/resnext resnext
#sh run.sh 2 10 64 8     _timeline/resnext resnext
#sh run.sh 2 10 64 16    _timeline/resnext resnext

#sh run.sh 2 10 128 2    _timeline/resnext resnext
#sh run.sh 2 10 128 4    _timeline/resnext resnext
#sh run.sh 2 10 128 8    _timeline/resnext resnext
#sh run.sh 2 10 128 16   _timeline/resnext resnext

#sh run.sh 4 10 16 2     _timeline/resnext resnext
#sh run.sh 4 10 16 4     _timeline/resnext resnext
#sh run.sh 4 10 16 8     _timeline/resnext resnext
#sh run.sh 4 10 16 16    _timeline/resnext resnext

#sh run.sh 4 10 32 2     _timeline/resnext resnext
#sh run.sh 4 10 32 4     _timeline/resnext resnext
#sh run.sh 4 10 32 8     _timeline/resnext resnext
#sh run.sh 4 10 32 16    _timeline/resnext resnext

#sh run.sh 4 10 64 2     _timeline/resnext resnext
#sh run.sh 4 10 64 4     _timeline/resnext resnext
#sh run.sh 4 10 64 8     _timeline/resnext resnext
#sh run.sh 4 10 64 16    _timeline/resnext resnext

#sh run.sh 4 10 128 2    _timeline/resnext resnext
#sh run.sh 4 10 128 4    _timeline/resnext resnext
#sh run.sh 4 10 128 8    _timeline/resnext resnext
#sh run.sh 4 10 128 16   _timeline/resnext resnext

#sh run.sh 8 10 16 2     _timeline/resnext resnext
#sh run.sh 8 10 16 4     _timeline/resnext resnext
#sh run.sh 8 10 16 8     _timeline/resnext resnext
#sh run.sh 8 10 16 16    _timeline/resnext resnext

#sh run.sh 8 10 32 2     _timeline/resnext resnext
#sh run.sh 8 10 32 4     _timeline/resnext resnext
#sh run.sh 8 10 32 8     _timeline/resnext resnext
#sh run.sh 8 10 32 16    _timeline/resnext resnext

#sh run.sh 8 10 64 2     _timeline/resnext resnext
#sh run.sh 8 10 64 4     _timeline/resnext resnext
#sh run.sh 8 10 64 8     _timeline/resnext resnext
#sh run.sh 8 10 64 16    _timeline/resnext resnext

#sh run.sh 8 10 128 2    _timeline/resnext resnext
#sh run.sh 8 10 128 4    _timeline/resnext resnext
#sh run.sh 8 10 128 8    _timeline/resnext resnext
#sh run.sh 8 10 128 16   _timeline/resnext resnext

#sh runh.sh 2 10 16 2     _timeline/resnext_h resnext
#sh runh.sh 2 10 16 4     _timeline/resnext_h resnext
#sh runh.sh 2 10 16 8     _timeline/resnext_h resnext
#sh runh.sh 2 10 16 16    _timeline/resnext_h resnext

#sh runh.sh 2 10 32 2     _timeline/resnext_h resnext
#sh runh.sh 2 10 32 4     _timeline/resnext_h resnext

#python test_single.py -md resnext -bs 1 -lgd single/resnext_bs1/
#tar zcvf _single/resnext_bs1.tgz single/resnext_bs1/
#rm -r single/resnext_bs1/

#python test_single.py -md resnext -bs 2 -lgd single/resnext_bs2/
#tar zcvf _single/resnext_bs2.tgz single/resnext_bs2/
#rm -r single/resnext_bs2/

#python test_single.py -md resnext -bs 4 -lgd single/resnext_bs4/
#tar zcvf _single/resnext_bs4.tgz single/resnext_bs4/
#rm -r single/resnext_bs4/

#python test_single.py -md resnext -bs 8 -lgd single/resnext_bs8/
#tar zcvf _single/resnext_bs8.tgz single/resnext_bs8/
#rm -r single/resnext_bs8/

#python test_single.py -md resnext -bs 16 -lgd single/resnext_bs16/
#tar zcvf _single/resnext_bs16.tgz single/resnext_bs16/
#rm -r single/resnext_bs16/

sh runh.sh 2 10 32 8     _timeline/resnext_h resnext
sh runh.sh 2 10 32 16    _timeline/resnext_h resnext

sh runh.sh 2 10 64 2     _timeline/resnext_h resnext
sh runh.sh 2 10 64 4     _timeline/resnext_h resnext
sh runh.sh 2 10 64 8     _timeline/resnext_h resnext
sh runh.sh 2 10 64 16    _timeline/resnext_h resnext

sh runh.sh 2 10 128 2    _timeline/resnext_h resnext
sh runh.sh 2 10 128 4    _timeline/resnext_h resnext
sh runh.sh 2 10 128 8    _timeline/resnext_h resnext
sh runh.sh 2 10 128 16   _timeline/resnext_h resnext

sh runh.sh 4 10 16 2     _timeline/resnext_h resnext
sh runh.sh 4 10 16 4     _timeline/resnext_h resnext
sh runh.sh 4 10 16 8     _timeline/resnext_h resnext
sh runh.sh 4 10 16 16    _timeline/resnext_h resnext

sh runh.sh 4 10 32 2     _timeline/resnext_h resnext
sh runh.sh 4 10 32 4     _timeline/resnext_h resnext
sh runh.sh 4 10 32 8     _timeline/resnext_h resnext
sh runh.sh 4 10 32 16    _timeline/resnext_h resnext

sh runh.sh 4 10 64 2     _timeline/resnext_h resnext
sh runh.sh 4 10 64 4     _timeline/resnext_h resnext
sh runh.sh 4 10 64 8     _timeline/resnext_h resnext
sh runh.sh 4 10 64 16    _timeline/resnext_h resnext

sh runh.sh 4 10 128 2    _timeline/resnext_h resnext
sh runh.sh 4 10 128 4    _timeline/resnext_h resnext
sh runh.sh 4 10 128 8    _timeline/resnext_h resnext
sh runh.sh 4 10 128 16   _timeline/resnext_h resnext

sh runh.sh 8 10 16 2     _timeline/resnext_h resnext
sh runh.sh 8 10 16 4     _timeline/resnext_h resnext
sh runh.sh 8 10 16 8     _timeline/resnext_h resnext
sh runh.sh 8 10 16 16    _timeline/resnext_h resnext

sh runh.sh 8 10 32 2     _timeline/resnext_h resnext
sh runh.sh 8 10 32 4     _timeline/resnext_h resnext
sh runh.sh 8 10 32 8     _timeline/resnext_h resnext
sh runh.sh 8 10 32 16    _timeline/resnext_h resnext

sh runh.sh 8 10 64 2     _timeline/resnext_h resnext
sh runh.sh 8 10 64 4     _timeline/resnext_h resnext
sh runh.sh 8 10 64 8     _timeline/resnext_h resnext
sh runh.sh 8 10 64 16    _timeline/resnext_h resnext

sh runh.sh 8 10 128 2    _timeline/resnext_h resnext
sh runh.sh 8 10 128 4    _timeline/resnext_h resnext
sh runh.sh 8 10 128 8    _timeline/resnext_h resnext
sh runh.sh 8 10 128 16   _timeline/resnext_h resnext

sh run_v.sh 2 10 16 2     _timeline/vgg_3mp vgg 
sh run_v.sh 2 10 16 4     _timeline/vgg_3mp vgg 
sh run_v.sh 2 10 16 8     _timeline/vgg_3mp vgg 
sh run_v.sh 2 10 16 16    _timeline/vgg_3mp vgg 

sh run_v.sh 2 10 32 2     _timeline/vgg_3mp vgg 
sh run_v.sh 2 10 32 4     _timeline/vgg_3mp vgg 
sh run_v.sh 2 10 32 8     _timeline/vgg_3mp vgg 
sh run_v.sh 2 10 32 16    _timeline/vgg_3mp vgg 

sh run_v.sh 2 10 64 2     _timeline/vgg_3mp vgg 
sh run_v.sh 2 10 64 4     _timeline/vgg_3mp vgg 
sh run_v.sh 2 10 64 8     _timeline/vgg_3mp vgg 
sh run_v.sh 2 10 64 16    _timeline/vgg_3mp vgg 

sh run_v.sh 2 10 128 2    _timeline/vgg_3mp vgg 
sh run_v.sh 2 10 128 4    _timeline/vgg_3mp vgg 
sh run_v.sh 2 10 128 8    _timeline/vgg_3mp vgg 
sh run_v.sh 2 10 128 16   _timeline/vgg_3mp vgg 

sh run_v.sh 4 10 16 2     _timeline/vgg_3mp vgg 
sh run_v.sh 4 10 16 4     _timeline/vgg_3mp vgg 
sh run_v.sh 4 10 16 8     _timeline/vgg_3mp vgg 
sh run_v.sh 4 10 16 16    _timeline/vgg_3mp vgg 

sh run_v.sh 4 10 32 2     _timeline/vgg_3mp vgg 
sh run_v.sh 4 10 32 4     _timeline/vgg_3mp vgg 
sh run_v.sh 4 10 32 8     _timeline/vgg_3mp vgg 
sh run_v.sh 4 10 32 16    _timeline/vgg_3mp vgg 

sh run_v.sh 4 10 64 2     _timeline/vgg_3mp vgg 
sh run_v.sh 4 10 64 4     _timeline/vgg_3mp vgg 
sh run_v.sh 4 10 64 8     _timeline/vgg_3mp vgg 
sh run_v.sh 4 10 64 16    _timeline/vgg_3mp vgg 

sh run_v.sh 4 10 128 2    _timeline/vgg_3mp vgg 
sh run_v.sh 4 10 128 4    _timeline/vgg_3mp vgg 
sh run_v.sh 4 10 128 8    _timeline/vgg_3mp vgg 
sh run_v.sh 4 10 128 16   _timeline/vgg_3mp vgg 

sh run_v.sh 8 10 16 2     _timeline/vgg_3mp vgg 
sh run_v.sh 8 10 16 4     _timeline/vgg_3mp vgg 
sh run_v.sh 8 10 16 8     _timeline/vgg_3mp vgg 
sh run_v.sh 8 10 16 16    _timeline/vgg_3mp vgg 

sh run_v.sh 8 10 32 2     _timeline/vgg_3mp vgg 
sh run_v.sh 8 10 32 4     _timeline/vgg_3mp vgg 
sh run_v.sh 8 10 32 8     _timeline/vgg_3mp vgg 
sh run_v.sh 8 10 32 16    _timeline/vgg_3mp vgg 

sh run_v.sh 8 10 64 2     _timeline/vgg_3mp vgg 
sh run_v.sh 8 10 64 4     _timeline/vgg_3mp vgg 
sh run_v.sh 8 10 64 8     _timeline/vgg_3mp vgg 
sh run_v.sh 8 10 64 16    _timeline/vgg_3mp vgg 

sh run_v.sh 8 10 128 2    _timeline/vgg_3mp vgg 
sh run_v.sh 8 10 128 4    _timeline/vgg_3mp vgg 
sh run_v.sh 8 10 128 8    _timeline/vgg_3mp vgg 
sh run_v.sh 8 10 128 16   _timeline/vgg_3mp vgg 

sh run.sh 2 10 16 2     _timeline/resnext resnext
sh run.sh 2 10 16 4     _timeline/resnext resnext
sh run.sh 2 10 16 8     _timeline/resnext resnext
sh run.sh 2 10 16 16    _timeline/resnext resnext

sh run.sh 2 10 32 2     _timeline/resnext resnext
sh run.sh 2 10 32 4     _timeline/resnext resnext
sh run.sh 2 10 32 8     _timeline/resnext resnext
sh run.sh 2 10 32 16    _timeline/resnext resnext

sh run.sh 2 10 64 2     _timeline/resnext resnext
sh run.sh 2 10 64 4     _timeline/resnext resnext
sh run.sh 2 10 64 8     _timeline/resnext resnext
sh run.sh 2 10 64 16    _timeline/resnext resnext

sh run.sh 2 10 128 2    _timeline/resnext resnext
sh run.sh 2 10 128 4    _timeline/resnext resnext
sh run.sh 2 10 128 8    _timeline/resnext resnext
sh run.sh 2 10 128 16   _timeline/resnext resnext

sh run.sh 4 10 16 2     _timeline/resnext resnext
sh run.sh 4 10 16 4     _timeline/resnext resnext
sh run.sh 4 10 16 8     _timeline/resnext resnext
sh run.sh 4 10 16 16    _timeline/resnext resnext

sh run.sh 4 10 32 2     _timeline/resnext resnext
sh run.sh 4 10 32 4     _timeline/resnext resnext
sh run.sh 4 10 32 8     _timeline/resnext resnext
sh run.sh 4 10 32 16    _timeline/resnext resnext

sh run.sh 4 10 64 2     _timeline/resnext resnext
sh run.sh 4 10 64 4     _timeline/resnext resnext
sh run.sh 4 10 64 8     _timeline/resnext resnext
sh run.sh 4 10 64 16    _timeline/resnext resnext

sh run.sh 4 10 128 2    _timeline/resnext resnext
sh run.sh 4 10 128 4    _timeline/resnext resnext
sh run.sh 4 10 128 8    _timeline/resnext resnext
sh run.sh 4 10 128 16   _timeline/resnext resnext

sh run.sh 8 10 16 2     _timeline/resnext resnext
sh run.sh 8 10 16 4     _timeline/resnext resnext
sh run.sh 8 10 16 8     _timeline/resnext resnext
sh run.sh 8 10 16 16    _timeline/resnext resnext

sh run.sh 8 10 32 2     _timeline/resnext resnext
sh run.sh 8 10 32 4     _timeline/resnext resnext
sh run.sh 8 10 32 8     _timeline/resnext resnext
sh run.sh 8 10 32 16    _timeline/resnext resnext

sh run.sh 8 10 64 2     _timeline/resnext resnext
sh run.sh 8 10 64 4     _timeline/resnext resnext
sh run.sh 8 10 64 8     _timeline/resnext resnext
sh run.sh 8 10 64 16    _timeline/resnext resnext

sh run.sh 8 10 128 2    _timeline/resnext resnext
sh run.sh 8 10 128 4    _timeline/resnext resnext
sh run.sh 8 10 128 8    _timeline/resnext resnext
sh run.sh 8 10 128 16   _timeline/resnext resnext

