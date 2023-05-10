First generate the thicklines data using the following command:

```
python generate_thicklines.py
```

## Generate Reverse Time Continuation (RTC) dataset

To generate the RTC dataset, use the following command:

```
python generate_data_rtc.py \
  --device_id 0 \
  --medium_type gaussian_lens
```

```
python generate_data_rtc.py \
  --device_id 0 \
  --medium_type gaussian_random_field
```

Here, each `device_id` corresponds to a GPU. If you have multiple GPUs, you can use different `device_id` in the above commands in parallel to generate the data faster.

## Generate Inverse Source (IS) dataset

TODO