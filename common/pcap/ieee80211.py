import struct


IE_ID_NAME_MAPPING = {
    0: 'SSID',
    1: 'Supported rates',
    2: 'FH Parameter Set',
    3: 'DS Parameter Set',
    4: 'CF Parameter Set',
    5: 'TIM',
    6: 'IBSS Parameter Set',
    7: 'Country',
    8: 'Hopping Pattern Parameters',
    9: 'Hopping Pattern Table',
    10: 'Request',
    11: 'BSS Load',
    12: 'EDCA Parameter Set',
    13: 'TSPEC',
    14: 'TCLAS',
    15: 'Schedule',
    16: 'Challenge text',
    32: 'Power Constraint',
    33: 'Power Capability',
    34: 'TPC Request',
    35: 'TPC Report',
    36: 'Supported Channels',
    37: 'Channel Switch Announcement',
    38: 'Measurement Request',
    39: 'Measurement Report',
    40: 'Quiet',
    41: 'IBSS DFS',
    42: 'ERP Information',
    43: 'TS Delay',
    44: 'TCLAS Processing',
    46: 'QoS Capability',
    48: 'RSN',
    50: 'Extended Supported Rates',
    127: 'Extended Capabilities',
    221: 'Vendor Specific',
}

IE_ID_LIST = sorted(list(IE_ID_NAME_MAPPING.keys()))

def macstr(macbytes):
    return ':'.join(['%02x' % k for k in macbytes])


def resolve_mac_type(mac):
    fc = mac.get('fc', 0)
    type = (fc >> 2) & 0x3
    subtype = (fc >> 4) & 0x0f
    return type, subtype


def is_blkack(mac):
    type, subtype = resolve_mac_type(mac)
    # control frame and block ack
    return type == 1 and subtype == 0x9


def is_qos_data(mac):
    type, subtype = resolve_mac_type(mac)
    return type == 2 and subtype == 0x8


def is_qos_null(mac):
    type, subtype = resolve_mac_type(mac)
    return type == 2 and subtype == 0xc


def is_qos(mac):
    return is_qos_null(mac) or is_qos_data(mac)


def translate_ie_id(id):
    if id in IE_ID_NAME_MAPPING:
        return IE_ID_NAME_MAPPING[id]
    else:
        return 'Reserved'


def ieee80211_mac_parse(packet, offset):
    hdr_fmt = "<HH6s"
    hdr_len = struct.calcsize(hdr_fmt)

    if len(packet) - offset < hdr_len:
        return 0, {}

    fc, duration, addr1 = \
        struct.unpack_from(hdr_fmt, packet, offset)

    offset += hdr_len
    mac = {
        'fc': fc,
        'duration': duration * .001024,
        'addr1': macstr(addr1),
    }

    if is_blkack(mac):
        blkack_fmt = "<6sHH8s"
        blkack_len = struct.calcsize(blkack_fmt)
        if len(packet) - offset < blkack_len:
            return offset, mac

        addr2, ba_ctrl, ba_ssc, ba_bitmap = \
            struct.unpack_from(blkack_fmt, packet, offset)
        offset += blkack_len
        mac.update({
            'addr2': macstr(addr2),
            'ba_ctrl': ba_ctrl,
            'ba_ssc': ba_ssc,
            'ba_bitmap': ba_bitmap
        })
        return offset, mac

    three_addr_fmt = "<6s6sH"
    three_addr_len = struct.calcsize(three_addr_fmt)
    if len(packet) - offset < three_addr_len:
        return offset, mac

    addr2, addr3, seq = \
        struct.unpack_from(three_addr_fmt, packet, offset)
    offset += three_addr_len
    mac.update({
        'addr2': macstr(addr2),
        'addr3': macstr(addr3),
        'seq': seq >> 4,
        'frag': seq & 0x0f
    })

    if is_qos(mac):
        four_addr_fmt = "<6s"
        four_addr_len = struct.calcsize(four_addr_fmt)
        if len(packet) - offset < four_addr_len:
            return offset, mac

        addr4, = struct.unpack_from(four_addr_fmt, packet, offset)
        offset += four_addr_len
        mac.update({
            'addr4': macstr(addr4)
        })

        qos_ctrl_fmt = "<H"
        qos_ctrl_len = struct.calcsize(qos_ctrl_fmt)
        if len(packet) - offset < qos_ctrl_len:
            return offset, mac

        qos_ctrl, = struct.unpack_from(qos_ctrl_fmt, packet, offset)
        tid = qos_ctrl & 0xf
        eosp = (qos_ctrl >> 4) & 1
        mesh_ps = (qos_ctrl >> 9) & 1
        rspi = (qos_ctrl >> 10) & 1

        mac.update({
            'tid': tid,
            'eosp': eosp,
            'rspi': rspi,
            'mesh_ps': mesh_ps,
        })

    return offset, mac


def ieee80211_mgt_parse(packet, offset=0, skip_last_len=4):
    tag_params = dict()
    while offset < len(packet) - skip_last_len -1:  # exclude checksum
        tag_id = packet[offset]
        tag_len = packet[offset + 1]
        offset += 2
        if offset + tag_len - 1 < len(packet) - skip_last_len:
            tag_data = packet[offset:offset + tag_len]
        else:
            tag_data = []
        if tag_id not in tag_params:
            tag_params[tag_id] = bytearray()
        tag_params[tag_id].extend(tag_data)
        offset += tag_len
    return offset, tag_params



