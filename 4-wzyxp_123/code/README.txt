About using the docker.

You can use following command to build docker image when you locate in ***/code/ with the DockerFile
sudo nvidia-docker build -t cdc:1.0 .

After that, you can test or train.

For Testing,
1. For example, 
2. Create a global wdata folder to put test or training results, note as /global/wdata 
3. Run following command:
    sudo nvidia-docker run -v /global/origin_data:/data:ro -v /global/wdata:/wdata -it cdc:1.0 sh test.sh
4. After the docker run finished, you can find solution.csv at /global/wdata/

For Training,
1. Similar as Testing, your traini.csv and test.csv should be both in a global directory, noted as /global/origin_data, since it will test after training directly.
2. Create a global wdata folder to put test or training results, note as /global/wdata 
3. Run following command:
    sudo nvidia-docker run -v /global/origin_data:/data:ro -v /global/wdata:/wdata -it cdc:1.0 sh train.sh
4. After the docker run finished, you can find 1.zip, 2.zip .... and 8.zip at /global/wdata/, they are can replace the trained models wgeted from google drive during testing.
5. You also can find solution.csv at /global/wdata, which is predicted by just trained models.
