import csv
from scapy.all import *

# location of pcap files
transmitters = {
	"ap_48f8-db07-0c20_36": ["ap_48f8-db07-0c20_36_2023-06-15_09-57-47.pcap","ap_48f8-db07-0c20_36_2023-06-15_10-01-02.pcap","ap_48f8-db07-0c20_36_2023-06-15_10-04-21.pcap","ap_48f8-db07-0c20_36_2023-06-15_10-07-36.pcap","ap_48f8-db07-0c20_36_2023-06-15_10-10-49.pcap","ap_48f8-db07-0c20_36_2023-06-15_10-14-05.pcap","ap_48f8-db07-0c20_36_2023-06-15_10-17-21.pcap","ap_48f8-db07-0c20_36_2023-06-15_10-20-45.pcap","ap_48f8-db07-0c20_36_2023-06-15_10-24-00.pcap","ap_48f8-db07-0c20_36_2023-06-15_10-27-19.pcap"], 
	"ap_48f8-db07-08a0_36": ["ap_48f8-db07-08a0_36_2023-06-15_09-57-48.pcap","ap_48f8-db07-08a0_36_2023-06-15_10-01-03.pcap","ap_48f8-db07-08a0_36_2023-06-15_10-04-21.pcap","ap_48f8-db07-08a0_36_2023-06-15_10-07-36.pcap","ap_48f8-db07-08a0_36_2023-06-15_10-10-50.pcap","ap_48f8-db07-08a0_36_2023-06-15_10-14-06.pcap","ap_48f8-db07-08a0_36_2023-06-15_10-17-21.pcap","ap_48f8-db07-08a0_36_2023-06-15_10-20-46.pcap","ap_48f8-db07-08a0_36_2023-06-15_10-24-01.pcap","ap_48f8-db07-08a0_36_2023-06-15_10-27-19.pcap"], 
	"ap_48f8-db10-c020_36": ["ap_48f8-db10-c020_36_2023-06-15_09-57-48.pcap","ap_48f8-db10-c020_36_2023-06-15_10-01-03.pcap","ap_48f8-db10-c020_36_2023-06-15_10-04-21.pcap","ap_48f8-db10-c020_36_2023-06-15_10-07-36.pcap","ap_48f8-db10-c020_36_2023-06-15_10-10-50.pcap","ap_48f8-db10-c020_36_2023-06-15_10-14-06.pcap","ap_48f8-db10-c020_36_2023-06-15_10-17-21.pcap","ap_48f8-db10-c020_36_2023-06-15_10-20-46.pcap","ap_48f8-db10-c020_36_2023-06-15_10-24-01.pcap","ap_48f8-db10-c020_36_2023-06-15_10-27-19.pcap"]
}
transmitters = {"entrance": ["entrance.pcapng"]}
# transmitters = {"ap_48f8-db07-0c20_36": ["ap_48f8-db07-0c20_36_2023-06-15_09-57-47.pcap"], "ap_48f8-db07-08a0_36": ["ap_48f8-db07-08a0_36_2023-06-15_09-57-48.pcap"], "ap_48f8-db10-c020_36": ["ap_48f8-db10-c020_36_2023-06-15_09-57-48.pcap"]}

unfiltered = True
channel_filter = 36

request_lists = []
# add all macaddress and seq to a list
for transmitter in transmitters:
	request_list = []
	for file in transmitters[transmitter]:
		# pcap = rdpcap(f"pcapfiles/{file}")
		pcap_reader = PcapReader(f"pcapfiles/{file}")
		
		# counter = 0
		# limit = 1000
		
		# while counter <= limit:
			
		# Loop through each packet in the pcap file
		while True:
			# counter += 1
			try:
				packet = pcap_reader.read_packet()
			except EOFError:
				# end of file
				break
			
			if packet.haslayer(Dot11ProbeReq):
				packet_type = "PROBE_REQUEST"
			elif packet.haslayer(Dot11ProbeResp):
				packet_type = "PROBE_RESPONSE"
				continue
			else:
				continue
				# packet_type = "nan"
			
			# unix timestamp
			timestamp = int(packet.time*1000)
			
			# 2.4 or 5 ghz
			frequency = packet[RadioTap].Channel
			if packet.haslayer(Dot11):
				if 2400 <= frequency <= 2500:
					radio_band = "2"
					base = 2407
				elif 5150 <= frequency <= 5850:
					radio_band = "5"
					base = 5000
				else:
					radio_band = "nan"
			
			# https://stackoverflow.com/questions/60473359/scapy-get-set-frequency-or-channel-of-a-packet
			# 2.4 and 5Ghz channels increment by 5
			ch = (frequency - base) // 5
			if ch != channel_filter and not unfiltered:
				continue
			
			# remove fragment number to get sequence number
			seq = packet.SC >> 4
					
			# mac address and formatting
			tx_addr = packet.addr2.replace(":", "").upper()
			
			# make a temp id with tx_addr and a four digit seq for request list later
			mac_seq = f"{tx_addr}_{str(seq).zfill(4)}"
					
			# check virtual bit todo
			is_virtual_mac = (int(tx_addr, 16) & (1 << 7)) != 0
			if is_virtual_mac and not unfiltered:
				continue
			
			# info element formatting
			info_element = bytes(packet.getlayer(Dot11Elt)).hex()
			
			# list of rssi from sensors
			rssi_dict = {}
			if packet.haslayer(RadioTap):
				rssi_dict.update({transmitter: packet[RadioTap].dBm_AntSignal})
				# rssi_dict.update({transmitter: f"{packet[RadioTap].dBm_AntSignal}, {file}"})
			
			# x y coordinates?
			x = "nan"
			y = "nan"
			
			# make a id with tx_addr and a four digit seq
			mac_seq = f"{tx_addr}_{str(seq).zfill(4)}"
			request_list.append({"mac_seq": mac_seq, "timestamp": timestamp, "type": packet_type, "radio_band": radio_band, "ch": ch, "seq": seq, "tx_addr": tx_addr, "is_virtual_mac": is_virtual_mac, "info_element": info_element, "rssi_dict": rssi_dict, "x": x, "y": y})
	
	request_lists.append(request_list)


request_dict = {}
# 10 seconds * 1000
timestamp_threshold = 10000

# add request for the other transmitters
for request_list in request_lists:
	for request in request_list:
		mac_seq = request["mac_seq"]
		while mac_seq in request_dict:
			# append mac_seq with a prime until you get a unique mac_seq
			if request["timestamp"] - request_dict[mac_seq]["timestamp"] > timestamp_threshold:
				# mac/seq has already been received but occured a long time ago
				mac_seq = mac_seq + "'"
			else:
				# mac/seq is from another sensor as timestamp is similar
				# copy existing request
				existing_request = request_dict[mac_seq].copy()
				# add new rssi to request rssi_dict
				existing_request["rssi_dict"].update(request["rssi_dict"])
				request_dict.update({mac_seq: existing_request})
				break
		else:
			# add new request to dict
			request_dict.update({mac_seq: request.copy()})


sorted_request_dict = dict(sorted(request_dict.items(), key=(lambda item: item[1]["timestamp"])))
	
# Open the CSV file for writing
with open("new_output_unfiltered.csv" if unfiltered else "new_output_filtered.csv", mode="w", newline="") as file:
	# dictwriter headers
	headers = ["timestamp", "type", "radio_band", "ch", "seq", "tx_addr", "is_virtual_mac", "info_element", "rssi_vec_len"]
	# add rssi columns for each transmitter
	for transmitter in transmitters:
		headers.append(f"rssi@{transmitter}")
	headers.extend(["x", "y"])
	
	writer = csv.DictWriter(file, fieldnames=headers)
	writer.writeheader()
	
				
	for item in sorted_request_dict.values():
		# add rssi columns for each transmitter
		for transmitter in transmitters:
			item.update({f"rssi@{transmitter}": item["rssi_dict"].get(transmitter, "nan")})
		# add rssi_vec_len
		item.update({"rssi_vec_len": len(item["rssi_dict"])})
		# item.update({"type": item["mac_seq"]})
		# remove rssi_dict and other temp values from item
		del item["rssi_dict"]
		del item["mac_seq"]
		
		writer.writerow(item)
