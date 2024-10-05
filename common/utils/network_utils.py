
LAA_SECOND_CHARS = ['2', '3', '6', '7', 'a', 'b', 'e', 'f']


def is_virtual_mac(mac_addr):
    return is_locally_admined_mac(mac_addr)


def is_locally_admined_mac(mac_addr):
    """
    Test whether the MAC address is locally administered address.
    :param mac_addr:
    :return:
    """
    if not isinstance(mac_addr, str):
        raise ValueError()
    mac_addr.replace(':', '').replace('-', '')
    mac_addr = mac_addr.lower()
    return mac_addr[2] in LAA_SECOND_CHARS


def mac_int_to_hexstr(int_mac):
    hex_mac = hex(int_mac)[2:]
    for _ in range(12 - len(hex_mac)):
        hex_mac = '0' + hex_mac
    return hex_mac


def mac_hexstr_to_int(hex_mac):
    return int(hex_mac, 16)


def frequency_to_channel(freq):
    if 2412 <= freq <= 2484:
        if freq == 2484:
            ch = 14
        else:
            ch = (freq - 2412) // 5 + 1
        return 2, ch
    elif 5150 <= freq <= 5925:
        ch = (freq - 5000) // 5
        return 5, ch


def channel_to_frequency(radio_band, ch):
    if radio_band == 2:
        return 2412 + 5 * (ch - 1)
    elif radio_band == 5:
        return 5000 + 5 * ch






