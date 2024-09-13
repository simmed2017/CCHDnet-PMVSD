Chinese Medical Named Entity Recognition
It aims to extract and categorize entities with specific meanings from unstructured text.

Chinese Medical Relation Extraction
It refers to recognizing specific entity pairs and their relations in the form of a triplet (subject, relation, object) from given texts.

Chinese Medical Entity Alignment
It aims at mapping non-standard clinical mentions, such as some fragments extracted from clinical texts, to standard terms within a certain knowledge base.


Example:

1. Chinese Medical Named Entity Recognition (MRC-CAP)
Input: Medical text
Output: Entities and their corresponding categories

Input text: "�Ҽ��ȱ��Ĥ�ܲ���8.3mm��"
Output: Entity 1: "�Ҽ��ȱ��", category: "Disease"; Entity 2: "Ĥ�ܲ�", category: "Location"
________________________________________

2. Chinese Medical Relation Extraction (CADA)
Input: Medical text
Output: Relational triplet (Subject, Relation, Object)

Input text: "�Ҽ��ȱ��Ĥ�ܲ���8.3mm��"
Output triplet: (�Ҽ��ȱ��Ĥ�ܲ���, size, 8.3mm)
________________________________________

3. Chinese Medical Entity Alignment (Med-GNN)
Input: Medical phrase, standard terminology database
Output: Corresponding standard terminology

Input medical phrase: "�Ҽ��ȱ��Ĥ�ܲ���"
standard terminology database: "Custom"
Output corresponding terminology: Ĥ�ܲ��Ҽ��ȱ��, converted to Flag2 (ventricular septal defect), encode: 1.