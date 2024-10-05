vendor_info_dict = {
    # vendor name: [[keywords],]
    'Cisco': [['cisco'],],
    'Realme': [['realme'],],
    'Intel': [['intel'],],
    'Nokia': [['nokia'],],
    'Aruba': [['aruba'],],
    'Dell': [['dell'],],
    'ZTE': [['zte'],],
    'Huawei': [['huawei'],],
    'Xiaomi': [['xiaomi'],],
    'Amazon': [['amazon'],],
    'Motorola': [['motorola'],],
    'vivo': [['vivo'],],
    'Samsung': [['samsung'],],
    'Nintendo': [['nintendo'],],
    'H3C': [['h3c'],],
    'China Mobile': [['china mobile'],],
    'Apple': [['apple'],],
    'HMD': [[' hmd '],],
    'Hon Hai': [['hon hai'],],
    'Tp-Link': [['tp-link'],],
    'Netgear': [['netgear'],],
    'Raspberry Pi': [['raspberry pi'],],
    'HP': [['hewlett packard'],],
    'LG': [[' lg ', 'lg '],],
    'OnePlus': [['oneplus'],],
    'Fujitsu': [['fujitsu'],],
    'Oppo': [['oppo'],],
    'Espressif': [['espressif'],],
    'Google': [['google'],],
    'Juniper': [['juniper'],],
    'NEC': [[' nec ', 'nec '],],
    'TI': [['texas instruments'],],
    'Microsoft': [['microsoft'],],
    'Siemens': [['siemens'],],
    'Lenovo': [['lenovo'],],
    'ASUS': [['asus'],],
    'Ubiquiti': [['ubiquiti'],],
    'Hikvision': [['hikvision'],],
    'HTC': [[' htc ', 'htc '],],
    'Ericsson': [['ericsson'],],
    'Sony': [['sony'],],
    'BlackBerry': [['blackberry'],],
    'Sharp': [['sharp'],],
    'Meizu': [['meizu'],],
    'Liteon': [['liteon'],]
}


def load_ouis(oui_filepath):
    with open(oui_filepath, 'r') as oui_file:
        oui_dict = {}
        for row in oui_file:
            if '(base 16)' in row:
                segs = row.split('(base 16)')
                oui = segs[0].strip().upper()
                vendor = segs[1].strip()
                oui_dict[oui] = vendor
    return oui_dict


def identify_oui(mac, oui_dict, simplified_vendor=True):
    oui = mac.replace(':', '').replace('-', '').upper()[:6]
    vendor_name = oui_dict.get(oui, 'Unknown')
    if simplified_vendor:
        vendor_name = simplify_vendor(vendor_name)
    return vendor_name


def simplify_vendor(vendor_fullname):
    for vendor_name, vendor_info in vendor_info_dict.items():
        keywords = vendor_info[0]
        for keyword in keywords:
            if keyword in vendor_fullname.lower():
                return vendor_name
    return vendor_fullname


if __name__ == '__main__':
    oui_filepath = ''
    oui_dict = load_ouis(oui_filepath)
    # print(oui_dict)
    print(identify_oui('6c5c145c1da4', oui_dict))
