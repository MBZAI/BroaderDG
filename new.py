
class GlobalAttentionBlock(nn.Module):
    def __init__(self):
        super(GlobalAttentionBlock, self).__init__()
        
    def forward(self, inputs):
        shape = inputs.size()
        x = F.avg_pool2d(inputs, kernel_size=(shape[2], shape[3]))
        x = nn.Conv2d(shape[1], shape[1], kernel_size=1, padding=0)(x)
        x = F.relu(x)
        x = nn.Conv2d(shape[1], shape[1], kernel_size=1, padding=0)(x)
        x = torch.sigmoid(x)
        C_A = torch.mul(x, inputs)
        
        x = torch.mean(C_A, dim=-1, keepdim=True)
        x = torch.sigmoid(x)
        S_A = torch.mul(x, C_A)
        return S_A

class CategoryAttentionBlock(nn.Module):
    def __init__(self, classes, k):
        super(CategoryAttentionBlock, self).__init__()
        self.classes = classes
        self.k = k
        self.conv = nn.Conv2d(in_channels=inputs.size()[1], out_channels=k*classes, kernel_size=1, padding=0)
        self.batch_norm = nn.BatchNorm2d(k*classes)
        
    def forward(self, inputs):
        shape = inputs.size()
        F = self.batch_norm(self.conv(inputs))
        F1 = F.relu(F)
        
        F2 = F1
        x = F.max_pool2d(F2, kernel_size=(shape[2], shape[3]))
        
        x = x.view(-1, self.classes, self.k)
        S = torch.mean(x, dim=-1, keepdim=False)
        
        x = F1.view(-1, shape[1], shape[2], self.classes, self.k)
        x = torch.mean(x, dim=-1, keepdim=False)
        x = torch.mul(S, x)
        M = torch.mean(x, dim=-1, keepdim=True)
        
        semantic = torch.mul(inputs, M)
        return semantic

class Biomedical_Clip_train_CAB(Biomedical_Clip_train):
    """
    Empirical Risk Minimization (ERM)
    """


    def __init__(self, input_shape, num_classes, num_domains, hparams, weights_for_balance):
        super(Biomedical_Clip_train_CAB, self).__init__(input_shape, num_classes, num_domains,
                                  hparams, weights_for_balance)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load BioBERT tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
        biobert_mlp_model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
        biobert_mlp_model = biobert_mlp_model.to(device)
        
        self.featurizer, preprocess = clip.load('ViT-B/16', device)
        self.featurizer=self.featurizer.float()

        
        self.GAB = GlobalAttentionBlock()
        self.CAB = CategoryAttentionBlock(classes= 5,k=5)

        self.network = nn.Sequential(self.featurizer, self.GAB, self.CAB)


        # if(self.hparams['weight_init']=="clip_full"):
        #     print("clip_full")
            # self.featurizer.network.proj=None

        # printNetworkParams(self.featurizer)
        self.optimizer = torch.optim.AdamW(
            list(self.network.visual.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.Class_names=["No DR","mild DR", "moderate DR", "severe DR", "proliferative DR"]
        
        # print("what is going.........................................")
        with torch.no_grad():
            sentences = [tokenizer(f"a photo of a {c}", return_tensors="pt")['input_ids'] for c in self.Class_names]
            # print(sentences)
            # pdb.set_trace()
            # text_inputs  = torch.cat(sentences)
            # text_inputs = text_inputs.to(device)
            # pdb.set_trace()
            # breakpoint()
            # print(sentences[0].shape)
            # print('----------'*300)
            # print(sentences[0].to(device))
            # print('hahahahaha'*300)
            # s = sentences[0]
            # s = s.to(device)
            # print(biobert_mlp_model(s))
            outputs = torch.cat([biobert_mlp_model(s.to(device)).pooler_output for s in sentences])
            # pdb.set_trace()
            self.text_features = outputs
            # del biobert_mlp_model
            # self.text_features = self.featurizer.encode_text(text_inputs)
        
        # print(self.Class_names)
        self.cnt=0

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        # with torch.no_grad():
       
        image_features = self.network.encode_image(all_x)
      
        # image_features = image_features @ self.featurizer.visual.proj
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = self.text_features / self.text_features.norm(dim=1, keepdim=True)
        logit_scale = self.network.logit_scale.exp()
        # print(image_features.shape, text_features.shape)
        logits_per_image = logit_scale * image_features @ text_features.t()
        loss=F.cross_entropy(logits_per_image, all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.cnt+=1
        return {'loss': loss.item()}
