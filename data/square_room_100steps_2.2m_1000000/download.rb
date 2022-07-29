#run this to download all the data 
(0..99).to_a.each do |x|
  num = format('%02d', x % 1000)
  url = "https://storage.googleapis.com/grid-cells-datasets/square_room_100steps_2.2m_1000000/00#{num}-of-0099.tfrecord"
  `wget #{url}`
end
